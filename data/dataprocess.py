# -*-coding:utf-8-*-

import re
import unicodedata
import pickle
import os

import sys

sys.path.append(os.path.abspath('..'))
from config import Config

config = Config()


# 字符处理类和数据准备类很大程度上参考了该博主的处理代码
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py

# 字符统计类
class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2count = {}  # 索引到词字典

        self.pad_idx = 0  # 填充字符索引位序
        self.unk_idx = 1  # 未知字符索引位序
        self.sos_idx = 2  # 开始字符索引位序
        self.eos_idx = 3  # 结束字符索引位序
        self.mask_idx = 4  # 屏蔽字符索引位序
        self.word2index = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3, '<mask>': 4}  # 字符到索引的字典
        self.index2word = {value:key for key, value in self.word2index.items()}

        self.n_words = 5  # 初始字典有5个字符
        self.seq_max_len = 0

        self.words = []  # 字典的全部字符

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('保留单词数 %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        # self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.index2word = {0: '<blank>', 1: '<unk>', 2: '<s>', 3: '</s>'}
        self.n_words = 4  # Count default tokens

        for word in keep_words:
            self.index_word(word)


# 数据准备类
class DataProcess(object):

    def __init__(self):
        self.src_max_len = 2  # 序列前后有SOS和EOS
        self.tgt_max_len = 2  # 序列前后有SOS和EOS

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.strip())
        s = re.sub(r"([,.!?])", r" \1 ", s)
        # s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
        s = re.sub(r"s+", r" ", s).strip()
        return s

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def filter_pairs(self, pairs):
        filtered_pairs = []
        for pair in pairs:
            sentence_num = 0
            for i in pair:
                print(len(i.split(' ')))
                if len(i.split(' ')) > config.min_len and len(i.split(' ')) <= config.max_len:
                    sentence_num += 1
            if sentence_num == len(pair):
                filtered_pairs.append(pair)
            else:
                temp = [len(i.split(' ')) for i in pair]
                print(temp, pair)

        return filtered_pairs

    def indexes_from_sentence(self, lang, sentence):
        # 前后加上sos和eos。注意句子的句号也要加上，如果这个词没有出现在词典中（已经去除次数小于限定的词），以unk填充
        if type(lang) == dict:
            return [config.sos] + \
                   [lang['word2index'].get(word, config.unk) for word in sentence.split(' ')] + \
                   [config.eos]
        else:
            return [config.sos] + \
                   [lang.word2index.get(word, config.unk) for word in sentence.split(' ')] + \
                   [config.eos]

    def read_file(self, data):
        content = open(data, encoding='utf-8').read().split('\n')  # 读取文件并处理
        # content = [self.normalize_string(s) for s in content]  # 规范化字符，并限制长度

        return content

    def process(self, corpus):
        """
        处理输入文件，并且返回数据类类对象
        :param corpus:
        :return:
        """
        contend = self.read_file(corpus)  # 读取并处理输入序列
        lang = Lang('corpus')

        for line in contend:
            lang.index_words(line)  # 检索序列中的单词
        lang.word2count = dict(sorted(lang.word2count.items(), key=lambda x: x[1], reverse=True))  # 降序排列word2count字典
        for key in lang.word2index.keys():  # 收集每一个具体的单词
            lang.words.append(key)

        return lang

    def save_vocab(self, vocab_obj):
        """
        高效的存储数据类对象
        :param vocab_obj: 数据类对象
        :return:
        """
        with open(config.vocab_file, 'wb') as f:
            pickle.dump(vocab_obj, f)

    def load_vocab(self, vocab_path):
        """
        读取数据存储文件，重新构造出数据类对象
        :param vocab_path: 数据类文件地址
        :return: 数据类对象
        """
        with open(vocab_path, 'rb') as f:
            return pickle.load(f)