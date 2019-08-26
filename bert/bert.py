# coding=utf-8

r"""
Bert模型主程序
"""

from .embeddings import BertEmbedding
from bert.transformer import TransformerEncoderLayer
from bert.utils import Mask

import torch
from torch import nn

from config import Config
config = Config()


class Bert(nn.Module):
    """
    Bert模型的主程序
    """
    def __init__(self, vocab_size, hidden_size):
        super(Bert, self).__init__()

        self.embedding = BertEmbedding(vocab_size, config.hidden_size)  # 输入表示向量层

        d_ff = hidden_size * 4

        self.transformer_block = nn.ModuleList(  # 堆叠若干层的transformer encoder层
            [TransformerEncoderLayer(d_ff, hidden_size) for _ in range(config.layers)])

        self.mask_obj = Mask()

    def forward(self, src_seq, segment):
        """

        :param src_seq:
        :param segment:
        :return:
        """
        input_emb = self.embedding(src_seq, segment)
        bert_in = input_emb

        pad_mask = self.mask_obj.padding_mask(src_seq, src_seq)
        no_pad_mask = self.mask_obj.no_padding_mask(src_seq)

        enc_out = None
        for layer in self.transformer_block:
            enc_out = layer(bert_in, pad_mask, no_pad_mask)
            bert_in = enc_out

        return enc_out





