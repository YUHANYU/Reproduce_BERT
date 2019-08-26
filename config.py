# coding=utf-8

r"""
BERT模型的参数类
"""

import torch

import os


class Config:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 设备
    # input_file = './data/sample.txt'  # 输入的原始文件（也可以是','分割的文件列表）
    # output_file = './data/result.txt'  # 输出文件路径（也可以是','分割的文件列表）
    # vocab_file = './vocab.txt'  # Bert模型训练过的词典文件
    do_lower_case = True  # True则忽略大小写，False执行小写
    max_seq_length = 128   # 每条训练数据（两条句子相加）后的最大长度限制
    max_predictions_per_seq = 20  # 每一条训练数据做mask的token的最大数量
    mask_p = 0.15  # 每条训练数据被mask的token的数量
    random_seed = 12345  # 随机种子数
    dupe_factor = 5  # 对输入文档多次重复随机产生训练集，随机产生的次数
    short_seq_prob = 0.1 # 以此概率产生小于最大序列长度的训练数据

    input_corpus = os.getcwd() + '/data/corpus.small'  # 输入文件
    vocab_file = os.getcwd() + '/vocab.small'  # 存储的系列化词典文件
    output_file = os.getcwd() + '/output.small'  # 输出的训练结果文件
    # infer_file = os.getcwd() + '/infer.txt'  # 推理数据集
    infer_file = None

    min_len = 1
    max_len = 128

    train_batch_size = 64  # 训练数据批次

    pad_idx = 0  # 填充字符索引位序
    unk_idx = 1  # 未知字符索引位序
    sos_idx = 2  # 开始字符索引位序
    eos_idx = 3  # 结束字符索引位序
    mask_idx = 4  # 屏蔽字符索引位序

    # base model
    d_model = 768
    hidden_size = 768
    layers = 12
    heads = 12
    d_q = 64
    d_k = 64
    d_v = 64

    dropout = 0.1

    lr = 0.001  # Adam优化器的学习率
    weight_decay = 0.01
    beta = 0.9

    warmup_steps = 8000

    IsNext = 1  # 句子B是句子A的实际下一句
    NotNext = 0  # 句子B不是句子A的实际下一句
    do_mask = 1  # 句子中的该token被mask
    not_mask = 0  # 句子中的该token没有被mask

    epochs = 100




