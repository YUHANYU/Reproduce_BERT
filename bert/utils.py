# coding=utf-8

r"""
bert模型的工具类
"""

import torch

import numpy as np
import math

from config import Config
config = Config()


# 特殊学习率优化器类
class SpecialOptimizer():

    def __init__(self, optimizer, warmup_steps, d_model, step_num=0):
        """
        随着训练步骤，学习率对应改变的模型优化器
        :param optimizer: 预定义的优化器
        :param warmup_steps: 预热步
        :param d_model: 模型维度
        :param step_num: 当前的模型的的训练步数，默认从0开始
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = step_num

    # 优化器梯度清零
    def zero_grad(self):
        self.optimizer.zero_grad()

    # 优化器更新学习率和步进
    def step_and_update_learning_rate(self):
        self.step_num += 1  # 批次训练一次，即为步数一次，自加1

        # 生成当前步的学习率
        lr = np.power(self.d_model, -0.5) * np.min([np.power(self.step_num, -0.5),
                                                    np.power(self.warmup_steps, -1.5) * self.step_num])

        # 把当前步的学习率赋值给优化器
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.step()

        return lr


# gelu()激活函数，区别于relu()激活函数
def gelu(x):
    """
    区别于relu()激活函数的gelu()激活函数
    :param x: 要激活的神经元
    :return:
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# 特殊学习率优化器类
class SpecialOptimizer():

    def __init__(self, optimizer, warmup_steps, d_model, step_num=0):
        """
        随着训练步骤，学习率对应改变的模型优化器
        :param optimizer: 预定义的优化器
        :param warmup_steps: 预热步
        :param d_model: 模型维度
        :param step_num: 当前的模型的的训练步数，默认从0开始
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = step_num

    # 优化器梯度清零
    def zero_grad(self):
        self.optimizer.zero_grad()

    # 优化器更新学习率和步进
    def step_and_update_learning_rate(self):
        self.step_num += 1  # 批次训练一次，即为步数一次，自加1

        # 生成当前步的学习率
        lr = np.power(self.d_model, -0.5) * np.min([np.power(self.step_num, -0.5),
                                                    np.power(self.warmup_steps, -1.5) * self.step_num])

        # 把当前步的学习率赋值给优化器
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.step()

        return lr


# Mask屏蔽符类
class Mask():

    def __init__(self):
        pass

    # 对序列做补齐的位置
    def padding_mask(self, seq_k, seq_q):
        """
        生成padding masking TODO 待解释seq_k和seq_q的关系和来源
        :param seq_k: B*L
        :param seq_q: B*L
        :return: 产生B*T*T的pad_mask输出
        """
        seq_len = seq_q.size(1)
        pad_mask = seq_k.eq(config.pad_idx)  # 通过比较产生pad_mask B*T

        return pad_mask.unsqueeze(1).expand(-1, seq_len, -1)

    # 不对序列做补齐的位置
    def no_padding_mask(self, seq):
        """
        pad_mask的反向操作
        :param seq: B*T
        :return: B*T*T
        """
        return seq.ne(config.pad_idx).type(torch.float).unsqueeze(-1)

    # 序列屏蔽，用于decoder的操作中
    def sequence_mask(self, seq):
        """
        屏蔽子序列信息，防止decoder能解读到，使用一个上三角形来进行屏蔽
        seq: B*T batch_size*seq_len
        :return: seq_mask B*T*T batch_size*seq_len*seq_len
        """
        batch_size, seq_len = seq.shape
        # 上三角矩阵来屏蔽不能看到的子序列
        seq_mask = torch.triu(
             torch.ones((seq_len, seq_len), device=config.device, dtype=torch.uint8), diagonal=1)

        return seq_mask.unsqueeze(0).expand(batch_size, -1, -1)