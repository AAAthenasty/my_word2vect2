# -*- coding: utf-8 -*-
import heapq
import random
from collections import defaultdict
from functools import total_ordering

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from tqdm.auto import tqdm


def sigmoid(x):
    return 1 - 1 / (1 + np.exp(x))


def load_pretrained_wv(wv_mat, word_list):
    #模型初始化时导入随机初始化的参数矩阵syn0和词语
    #载入训练完成的词向量，方便进行那些most_similar之类的计算
    """用来导入将提前训练好的词向量（一个numpy二维数组）导入到gensim的KeyedVectors中
    wv_mat是词向量矩阵（n*k），word_list为词语列表（长度为n）
    wv_mat的第i行要对应word_list的第i个词语
    """
    from gensim.models.keyedvectors import KeyedVectors, Vocab
    vocab_size, vector_size = wv_mat.shape#syn0
    wv = KeyedVectors(vector_size)
    wv.vector_size = vector_size
    wv.vectors = wv_mat
    wv.index2word = word_list
    for i, w in enumerate(word_list):
        wv.vocab[w] = Vocab(index=i, count=vocab_size - i)#？？
    return wv


@total_ordering
class WordHuffmanNode(object):

    def __init__(self, freq=0, word=None):
        self.left = None
        self.right = None
        self.parent = None
        self.word = word
        self.freq = freq

    @classmethod
    def new(cls, left, right):
        node = cls(left.freq + right.freq)
        node.left, node.right = left, right
        left.parent = right.parent = node
        return node

    @property
    def is_leaf(self):
        return self.word is not None

    @property
    def children(self):
        return (self.left, self.right)

    def __eq__(self, other):
        return self.freq == other.freq

    def __lt__(self, other):
        return self.freq < other.freq


class WordHuffmanTree(object):

    def __init__(self, word_count):
        self.word_nodes = {}
        self.h = []
        for word, freq in word_count.items():
            node = WordHuffmanNode(freq, word)
            self.word_nodes[word] = node
            heapq.heappush(self.h, node)

    def __getitem__(self, word):
        return self.word_nodes[word]

    def merge_nodes(self):
        if len(self.h) > 1:
            left, right = heapq.heappop(self.h), heapq.heappop(self.h)
            node = WordHuffmanNode.new(left, right)  # 合并两个节点创建新节点
            heapq.heappush(self.h, node)

    def build_tree(self):
        # 合并节点直至只剩下一个节点
        while len(self.h) > 1:
            self.merge_nodes()

        self.root_node = self.h[0]  # 根节点
        self.root_node.code = ""  # Huffman编码
        self.root_node.node_path = []  # 从根节点到当前节点所经历的节点的ID

        i = 0
        node_list = [self.root_node]
        while node_list:
            node = node_list.pop(0)
            node.node_id = i
            i += 1
            for j, child in enumerate(node.children):
                child.code = node.code + str(j)
                child.node_path = node.node_path + [node.node_id]
                if not child.is_leaf:
                    node_list.append(child)
        self.max_node_id = i - 1

        for word_node in self.word_nodes.values():
            word_node.code_array = np.array([int(x) for x in word_node.code], dtype=np.bool)


class CustomWord2Vec(object):

    def __init__(self, sentences, size=100, window=5, min_count=5,
                 sg=0, hs=0, negative=5, ns_exponent=0.75, batch_words=10000,
                 cbow_mean=0, alpha=0.025, min_alpha=0.0001, seed=10, iter=5):
        self.size = size#词向量维度
        self.window = window#窗口大小 
        self.min_count = min_count
        self.sg = sg# sg: 是0则是CBOW模型，是1则是Skip-Gram模型
        self.hs = hs#hs: 是0则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax
        self.negative = negative#使用Negative Sampling时负采样的个数，默认是5
        self.ns_exponent = ns_exponent#决定负采样采样时每个词w的线段长度，分子和分母都取了3/4次幂
        self.cbow_mean = cbow_mean#为1则为上下文的词向量的平均值
        self.alpha = alpha#随机梯度下降法中迭代的初始步长
        self.min_alpha = min_alpha#最小的迭代步长值
        self.seed = seed
        self.iter = iter
        self.batch_words = batch_words
        self.model_initialized = False

        if sentences is not None:
            self.train(sentences)

    def count_words(self, sentences):
        word_count = defaultdict(int)
        for sentence in sentences:
            for word in sentence:
                word_count[word] += 1
        #8) min_count:需要计算词向量的最小词频。
        #这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。
        #k是词，v是个数
        word_count = dict((k, v) for k, v in word_count.items() if v >= self.min_count)
        word_list = sorted(list(word_count.keys()))
        word2index_map = dict(zip(word_list, range(len(word_list))))
        total_words = sum(word_count.values())
        word_freq = [word_count[x] for x in word_list]
        threshold = 0.001 * total_words
        word_sample_int = np.sqrt(np.array(word_freq) / threshold) + 1
        word_sample_int *= threshold / np.array(word_freq) * 1e8
        word_sample_int = np.maximum(1e8 - word_sample_int, 0).astype(int)

        self.word_count = word_count
        self.word_list = word_list
        self.word_num = len(word_list)
        self.word2index_map = word2index_map
        self.total_words = total_words
        self.word_freq = word_freq
        self.word_sample_int = dict(zip(word_list, word_sample_int))

    def init_model(self, sentences):
        self.cur_lr = self.alpha
        self.counter = 0  # 计数器，每训练一个窗口就加一
        self.random = np.random.RandomState(self.seed)
        # 遍历一遍语料获取词表、词频等信息
        self.count_words(sentences)
        # 最终的词向量矩阵，就是Skip-Gram中的中心词词向量矩阵U或者CBOW中的上下文词词向量矩阵V
        # 初始化：[-1/2d, 1/2d]之间的均匀分布随机数生成一个（词语个数*词向量维度d）的矩阵
        self.syn0 = (self.random.rand(self.word_num, self.size) - 0.5) / self.size
        self.wv = load_pretrained_wv(self.syn0, self.word_list)

        if self.hs:
            # 构建霍夫曼树
            self.tree = WordHuffmanTree(self.word_count)
            self.tree.build_tree()

            # 非叶节点的二分类logistic模型参数：全0初始化
            self.syn1 = np.zeros((self.tree.max_node_id + 1, self.size))
        else:
            # 负采样词语——提前根据权重构造一个大型数组，之后再使用randint从里面采样得到词语
            # 主要是np.random.choice比np.random.randint慢太多了
            word_freq = np.array([self.word_count[w] for w in self.word_list])#分子
            word_weight = word_freq ** self.ns_exponent#加总起来是分母
            #负采样过程
            factor = 50
            if word_weight.sum() * factor > 1e8:
                factor = max(np.floor(1e8 / word_weight.sum()), 1)
            self.neg_table = np.repeat(range(self.word_num), (word_weight * factor).astype(int))
            self.neg_table_size = len(self.neg_table)

            # Skip-Gram中的上下文词词向量矩阵V或者CBOW中的中心词词向量矩阵V
            # 也相当于word_num个二分类logistic模型的参数：全0初始化
            self.syn1neg = np.zeros((self.word_num, self.size))

        self.model_initialized = True

    def update_lr(self):#迭代步长逐渐减小
        if self.counter % self.batch_words:
            return
        all_words = self.total_words * self.iter
        progress = self.counter / (all_words - self.batch_words)
        self.cur_lr = self.alpha - (self.alpha - self.min_alpha) * progress
        self.cur_lr = max(self.cur_lr, self.min_alpha)

    def train_epoch(self, sentences):
        for sentence in tqdm(sentences):
            new_sentence = []
            for w in sentence:
                if w not in self.word_count:
                    continue
                self.counter += 1
                self.update_lr()
                sample_int = self.word_sample_int[w]
                if not sample_int or sample_int <= self.random.randint(1e8):
                    new_sentence.append(self.word2index_map[w])
                    #word2index_map依次放入
            for i, center_word_id in enumerate(new_sentence):#每个词都做一遍中心词
                new_window = np.random.randint(self.window) + 1#c不是固定的，而是在1-window中随机取
                start = max(0, i - new_window)
                end = min(len(new_sentence), i + new_window + 1)
                context_word_idx = list(new_sentence[start: i] + new_sentence[(i + 1): end])#一般情况长度为2c
                if self.hs:
                    self.train_hs_word_pair(center_word_id, context_word_idx)
                elif self.negative:
                    self.train_negative_word_pair(center_word_id, context_word_idx)

    def train(self, sentences):
        if not self.model_initialized:
            self.init_model(sentences)
        if not self.hs and self.negative <= 0:
            return
        for epoch in tqdm(range(1, self.iter + 1)):
            self.train_epoch(sentences)

    def train_negative_word_pair(self, center_word_id, context_word_idx):
        if self.sg:#中心词去预测上下文词
            for context_word_id in context_word_idx:#遍历2c个上下文词
                syn0_center_word = self.syn0[center_word_id]#取出该正例词对应训练词（中心词）的beta_0
                #进行负采样，词语编号
                negative_word_idx = self.random.randint(self.neg_table_size, size=self.negative)
                negative_word_idx = self.neg_table[negative_word_idx]
                #（保证采样负例）
                negative_word_idx = [x for x in negative_word_idx if x != context_word_id]
                related_word_idx = [context_word_id] + list(set(negative_word_idx))
                #正例和neg放在一起，组成包含neg+1个词的该样本的集合
                syn1neg_related_words = self.syn1neg[related_word_idx]#取出他们的beta^(w_i)
                #提前定义了sigmoid函数
                score = sigmoid(syn1neg_related_words.dot(syn0_center_word))#f=  σ(x_(w_0)* beta^(w_i) )
                score[0] -= 1 #-g = -(y_j −f) 此时第一个j=0是正例，y_j是1，其余y_j是0，相当于不变

                grad_syn1neg_related_words = score.reshape((-1, 1)).dot(syn0_center_word.reshape((1, -1)))#-g*x_(w_0 )用来更新beta
                grad_syn0_center_word = score.dot(syn1neg_related_words)#-e = -g* beta^(w_i)
                #只更新了中心词词向量
                self.syn0[center_word_id] -= grad_syn0_center_word * self.cur_lr#x_(w_0i)+e
                #更新0-neg个词的参数
                self.syn1neg[related_word_idx] -= grad_syn1neg_related_words * self.cur_lr#减去-g*x_(w_0 )更新beta
        
        else:#cbow预测中心词
            context_word_idx = list(set(context_word_idx))
            syn0_context_words = self.syn0[context_word_idx].mean(0)#上下文词词向量求平均
            negative_word_idx = self.random.randint(self.neg_table_size, size=self.negative)
            negative_word_idx = self.neg_table[negative_word_idx]
            #采样负例词
            negative_word_idx = [x for x in negative_word_idx if x != center_word_id]
            #1和neg
            related_word_idx = [center_word_id] + list(set(negative_word_idx))
            syn1neg_related_words = self.syn1neg[related_word_idx]#(neg+1)*M

            score = sigmoid(syn1neg_related_words.dot(syn0_context_words))#(neg+1)*M ** M*1=(neg+1)*1
            score[0] -= 1

            grad_syn1neg_related_words = score.reshape((-1, 1)).dot(syn0_context_words.reshape(1, -1))
            #(neg+1)*1 ** 1*M = (neg+1)*M
            grad_syn0_context_words = score.dot(syn1neg_related_words)#M*1

            self.syn1neg[related_word_idx] -= grad_syn1neg_related_words * self.cur_lr
            self.syn0[context_word_idx] -= grad_syn0_context_words * self.cur_lr

    def train_hs_word_pair(self, center_word_id, context_word_idx):
        if self.sg:
            for context_word_id in context_word_idx:
                syn0_center_word = self.syn0[center_word_id]
                context_word_node = self.tree[self.word_list[context_word_id]]
                syn1_path = self.syn1[context_word_node.node_path]

                score = sigmoid(syn1_path.dot(syn0_center_word))
                score -= context_word_node.code_array

                grad_syn0_center_word = score.dot(syn1_path)
                grad_syn1_path = score.reshape((-1, 1)).dot(syn0_center_word.reshape(1, -1))

                self.syn0[center_word_id] -= grad_syn0_center_word * self.cur_lr
                self.syn1[context_word_node.node_path] -= grad_syn1_path * self.cur_lr
        else:
            context_word_idx = list(set(context_word_idx))
            syn0_context_words = self.syn0[context_word_idx].mean(0)
            center_word_node = self.tree[self.word_list[center_word_id]]
            syn1_path = self.syn1[center_word_node.node_path]

            score = sigmoid(syn1_path.dot(syn0_context_words))
            score = score - center_word_node.code_array

            grad_syn0_context_words = score.dot(syn1_path)
            grad_syn1_path = score.reshape((-1, 1)).dot(syn0_context_words.reshape(1, -1))

            self.syn0[context_word_idx] -= grad_syn0_context_words * self.cur_lr
            self.syn1[center_word_node.node_path] -= grad_syn1_path * self.cur_lr


class SVDWord2Vec(object):

    def __init__(self, sentences=None, size=100, window=5, min_count=5, negative=5):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        if sentences is not None:
            self.train(sentences)

    def count_words(self, sentences):
        wordpair_count = defaultdict(int)
        word_count = defaultdict(int)
        for sentence in sentences:
            for word in sentence:
                word_count[word] += 1
        word_count = dict((k, v) for k, v in word_count.items() if v >= self.min_count)
        word_list = sorted(list(word_count.keys()))
        word2index_map = dict(zip(word_list, range(len(word_list))))

        total_words = sum(word_count.values())
        for sentence in sentences:
            sentence = [w for w in sentence if w in word_count]
            for i in range(1, self.window + 1):
                for word1, word2 in zip(sentence[i:], sentence[:-i]):
                    wordpair_count[(word1, word2)] += 1
                    wordpair_count[(word2, word1)] += 1

        self.word_count = word_count
        self.wordpair_count = wordpair_count
        self.word_list = word_list
        self.word2index_map = word2index_map
        self.total_words = total_words

    def build_mat(self):
        mat_data = []
        row_idx = []
        col_idx = []

        word_count = self.word_count
        word2index_map = self.word2index_map
        wordpair_count = self.wordpair_count
        for (word1, word2), num in wordpair_count.items():
            if word1 not in word_count or word2 not in word_count:
                continue
            word1_id = word2index_map[word1]
            word1_count = word_count[word1]
            word2_id = word2index_map[word2]
            word2_count = word_count[word2]
            row_idx.append(word1_id)
            col_idx.append(word2_id)
            if self.negative <= 0:
                mat_data.append(np.log1p(num))
            else:
                d = np.log(num * self.total_words / word1_count / word2_count)
                d -= np.log(self.negative)
                mat_data.append(max(d, 0))

        self.mat = coo_matrix((mat_data, (row_idx, col_idx)))

    def train(self, sentences):
        self.count_words(sentences)
        self.build_mat()

        U, lambda0, V = svds(self.mat, k=self.size)
        self.wv_mat = U.dot(np.diag(np.sqrt(lambda0)))
        self.wv = load_pretrained_wv(self.wv_mat, self.word_list)
