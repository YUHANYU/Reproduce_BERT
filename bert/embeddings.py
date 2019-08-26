# coding=utf-8

r"""
Bert模型的输入表示向量程序
"""

import torch
from torch import nn

import numpy as np

from config import Config
config = Config()


# 位置编码向量类
class PositionEmbedding(nn.Module):
    r""" Positional Encoding 位置编码类说明
    由于没有使用传统的基于RNN或者CNN的结构，那么输入的序列就无法判断其顺序信息，但这序列来说是十分重要的，
    Positional Encoding的作用就是对序列中词语的位置进行编码，这样才能使模型学会顺序信息。
    使用正余弦函数编码位置，pos在偶数位置为正弦编码，奇数位置为余弦编码。
    PE(pos, 2i)=sin(pos/10000^(2i/d_model)
    PE(pos, 2i+1)=cos(pos/10000^2i/d_model)

    即给定词语位置，可以编码为d_model维的词向量，位置编码的每一个维度对应正弦曲线，
    上面表现出的是位置编码的绝对位置编码（即只能区别前后位置，不能区别前前...或者后后...的相对位置）
    又因为正余弦函数能表达相对位置信息，即:
    sin(a+b)=sin(a)*cos(b)+cos(a)*sin(b)
    cos(a+b)=cos(a)*cos(b)-sin(a)*sin(b)
    对于词汇之间的位置偏移k，PE(pos+k)可以表示成PE(pos)+PE(k)组合形式，
    那么就能表达相对位置（即能区分长距离的前后）。
    """
    def __init__(self, max_seq_len, d_model, pad_idx):
        """
        位置编码
        :param max_seq_len: 序列的最大长度
        :param d_model: 模型的维度
        :param pad_idx: 填充符位置，默认为0
        """
        super(PositionEmbedding, self).__init__()
        pos_enc = np.array([
            [pos / np.power(10000, 2.0 * (i // 2) / d_model) for i in range(d_model)]
            for pos in range(max_seq_len)])  # 构建位置编码表

        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # sin计算偶数位置
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # cos计算奇数位置

        pos_enc[pad_idx] = 0.  # 第一行默认为pad符，全部填充0

        self.pos_embedding = nn.Embedding(max_seq_len, d_model)  # 设置位置向量层 L*D
        # 载入位置编码表，并不更新位置编码层
        self.pos_embedding.from_pretrained(torch.FloatTensor(pos_enc), freeze=True)

    def forward(self, src_seq):
        # 返回该批序列每个序列的字符的位置编码embedding
        return self.pos_embedding(src_seq.to(device=config.device))
    

# 段落编码向量类
class SegmentEmbedding(nn.Module):
    """
    段落编码向量类
    """
    def __init__(self, emb_dim):
        """
        :param emb_dim: 段落向量维度大小
        """
        super(SegmentEmbedding, self).__init__()
        segment_num = 3  # 段落向量标记数 0：pad, 1:sent_A, 2:sent_B
        self.segment_emb = nn.Embedding(segment_num, emb_dim)

    def forward(self, segment):
        return self.segment_emb(segment).to(config.device)


# 字符编码向量类
class TokenEmbedding(nn.Module):
    """
    字符编码向量（即词向量）类
    """
    def __init__(self, vocab_size, emb_dim):
        """
        :param vocab_size: 字典词数
        :param emb_dim: 词向量维度
        """
        super(TokenEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=config.pad_idx)

    def forward(self, token):
        return self.token_emb(token).to(config.device)


# Bert模型的输入表示（向量）
class BertEmbedding(nn.Module):
    """
    Bert模型的输入表示(向量)类
    """
    def __init__(self, vocab_size, emb_dim):
        """
        初始化，构建三个向量矩阵
        :param vocab_size: 词典词数
        :param emb_dim: 向量维度
        :param dropout: dropout大小
        """
        super(BertEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, emb_dim)
        self.segment_embedding = SegmentEmbedding(emb_dim)
        self.position_embedding = PositionEmbedding(vocab_size, config.hidden_size, config.pad_idx)

        self.dropout = nn.Dropout(config.dropout)

        # TODO layer norm

    def forward(self, seq, segment_lab):
        """

        :param seq:
        :param segment_lab:
        :return:
        """
        token_emb = self.token_embedding(seq)  # token词向量
        pos_emb = self.position_embedding(seq)  # 位置向量
        seg_emb = self.segment_embedding(segment_lab)  # 段落向量

        bert_input = token_emb + pos_emb + seg_emb  # bert的输入表示是三个向量的叠加

        bert_input = self.dropout(bert_input)  # 做dropout

        return bert_input