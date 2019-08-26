# coding=utf-8

r"""
BERT模型的训练数据对象
"""

from torch.utils.data import Dataset

import torch

from tqdm import tqdm
import random

from config import Config
config = Config()

class BertData:
    """
    Bert模型数据类对象，处理训练数据，便于Bert模型训练
    """
    def __init__(self, train_corpus, vocab_obj):
        """
        Bert模型数据类程序初始函数
        :param train_corpus: 训练数据集
        :param vocab_obj: 词典对象
        """
        self.vocab_obj = vocab_obj  # 词典对象
        with open(train_corpus, 'r', encoding='utf-8') as file:  # 读取训练数据集
            self.contend = [line.split('\\t') for line in file]  # 读取并切分文件
            self.lines = len(self.contend)  # 总文件行数

    def build_mlm_nsp(self):
        """
        构建bert模型训练需要做的mask token和nsp的句子
        :return:
        """
        def masked_sentence_words(sent, sent_a=None):
            """
            屏蔽句子中15%的token
            :param sent: 目标句子
            :param sent_a: 是句子a还是句子b
            :return:
            """
            sentence_token2mask = []  # 记录句子每个token是否做mask，做了token就记录实际index；没做就记录为0，用来计算损失
            sentence_token2idx = []  # 转化为句子中每个token为对应的索引

            for idx, token in enumerate(sent.split()):
                token_mask_p = random.random()  # token被mask的概率
                if token_mask_p < 0.15:  # 当token被mask概率小于15%，代表此token被选中做mask
                    token_mask_type = token_mask_p / 0.15  # token做mask的类型

                    if token_mask_p < 0.8:  # 80%的时间token被mask符替代
                        sentence_token2idx.append(self.vocab_obj.mask_idx)

                    elif token_mask_type < 0.9:  # 10%的几率token被一个随机词替代
                        replace_token = random.randint(self.vocab_obj.words)
                        sentence_token2idx.append(self.vocab_obj.word2index(replace_token))

                    else:  # 10%的几率token保持原样
                        sentence_token2idx.append(self.vocab_obj.word2index(token))

                    sentence_token2mask.append(self.vocab_obj.word2index[token])  # 记录该token的实际索引

                else:  # 不需要遮蔽这个token
                    sentence_token2idx.append(self.vocab_obj.word2index[token])  # 转化该token为对应的索引
                    sentence_token2mask.append(config.not_mask)  # 记录该token没有被mask

            sentence_token2idx += [self.vocab_obj.eos_idx]  # token2idx后面加上eos符=相当于SEP
            sentence_token2mask += [self.vocab_obj.pad_idx]  # 因为token2idx后面加上EOS符，对应的token2mask后面也要加上pad符，表示屏蔽

            if sent_a:  # 如果是句子A
                sentence_token2idx = [self.vocab_obj.sos_idx] + sentence_token2idx  # 前面加上SOS符=CLS符
                sentence_token2mask = [self.vocab_obj.pad_idx] + sentence_token2mask  # 前面加上PAD符，表示屏蔽

            return sentence_token2mask, sentence_token2idx

        # next sentence prediction，二值化下一句预测任务，50%的句子B为句子A的下一句，50%句子B不是句子A的下一句
        sent_a_b = []  # 存储句子A和句子B
        for line in self.contend:  # 迭代构建句子a和句子b数据集
            sent_a = line[0]
            if random.random() < 0.5:  # 随机选择50%的句子为实际的下一句
                sent_b = line[1]
                sent_a_b.append([sent_a, sent_b, config.IsNext])
            else:  # 随机选择50%的句子不是实际的下一句，即从数据集中随机的选择依据为下一句
                sent_b = self.contend[random.randrange(len(self.contend))][1]
                sent_a_b.append([sent_a, sent_b, config.NotNext])

        token_mask_segment_nsp = []  # 列表记录bert模型的token-mask-segment-nsp
        # masked lm，屏蔽语言模型，每个句子的15%token被屏蔽
        for idx, line in enumerate(sent_a_b):  # 迭代构建待mask的句子a和句子b数据集
            token2mask_a, token2idx_a = masked_sentence_words(line[0], True)  # 对句子A做处理，得到句子A的token索引和mask
            token2mask_b, token2idx_b = masked_sentence_words(line[1])  # 对句子B做处理，得到句子B的token索引和mask

            segment_idx = ([1 for _ in range(len(token2idx_a))] + \
                           [2 for _ in range(len(token2idx_b))])[:config.max_predictions_per_seq]  # 句子分割符,1=A，2=B

            token_len = len(token2idx_a + token2idx_b)  # 输入字符索引长度
            mask_len = len(token2mask_a + token2mask_b)  # mask索引长度
            segment_len = len(segment_idx)  # 句子分割索引长度

            assert token_len == mask_len and token_len == segment_len and mask_len == token_len
            'BERT模型的输入token索引和句子索引以及屏蔽符索引三个序列不相等，无法计算！！！'

            bert_token = token2idx_a + token2idx_b + \
                         [self.vocab_obj.pad_idx for _  in range(config.max_predictions_per_seq - token_len)]  # 补齐最大长度
            bert_mask = token2mask_a + token2mask_b + \
                        [self.vocab_obj.pad_idx for _  in range(config.max_predictions_per_seq - mask_len)]  # 补齐最大长度
            bert_segment = segment_idx + \
                        [self.vocab_obj.pad_idx for _ in range(config.max_predictions_per_seq - segment_len)]  # 补齐最大长度
            bert_nsp = line[2]

            bert_mlm_nsp = {
                'token': bert_token,  # bert模型输入的token索引
                'mask': bert_mask,  # bert模型做masked lm的mask索引
                'segment': bert_segment,  # bert模型的segment索引
                'nsp': bert_nsp  # bert模型做nsp的下一句指示符
            }

            bert_mlm_nsp = {key: torch.tensor(value) for key, value in bert_mlm_nsp.items()}
            token_mask_segment_nsp.append(bert_mlm_nsp)

        return token_mask_segment_nsp






