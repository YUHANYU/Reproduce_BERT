# coding=utf-8

r"""
预训练bert的主程序
"""

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from data.dataprocess import DataProcess
from bert.bertdata import BertData
from bert.utils import SpecialOptimizer
from bert.pre_training_bert import PreTrainingBert
from bert.bert import Bert

from config import Config
config = Config()


def main():
    data_obj = DataProcess()  # 数据类对象
    vocab_obj = data_obj.process(config.input_corpus)  # 处理输入文件，生成词典对象
    data_obj.save_vocab(vocab_obj)  # 存储词典对象

    train_data_obj = BertData(config.input_corpus, vocab_obj)  # Bert数据类的对象

    train_data = train_data_obj.build_mlm_nsp()

    # TODO 加载推理数据类对象

    train_data_loader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)  # 打包训练数据做批次训练

    # TODO 打包推理数据类对象

    bert = Bert(vocab_obj.n_words, config.hidden_size)  # bert模型

    pre_training = PreTrainingBert(bert, vocab_obj.n_words, config.hidden_size)  # 预训练bert模型

    special_optim = SpecialOptimizer(  # 变化学习率的优化器
        optimizer=optim.Adam(
            params=pre_training.parameters(), lr=config.lr, weight_decay=config.weight_decay),
        warmup_steps=config.warmup_steps,
        d_model=config.hidden_size)

    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)  # 损失函数-负对数似然损失函数，忽略pad填充符

    pre_training.pre_training(special_optim, criterion, train_data_loader)


if __name__ == "__main__":
    main()








