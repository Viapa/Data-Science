# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: Tensor.py
@date: 2023/9/13 16:05
@target: 学会掌握 Pytorch 中的 Tensor 的相关基础用法
"""

import numpy as np
import torch


# 创建一个张量, 并指定其数据类型
def create_tensor(val, dtype):
    """
    tensor支持的数据类型包括:
    浮点型: torch.float64(torch.double), torch.float32(torch.float), torch.float16
    整型: torch.int64(torch.long), torch.int32(torch.int), torch.int16, torch.int8, torch.uint8
    布尔型: torch.bool
    不支持str类型
    """
    x = torch.tensor(val, dtype=dtype)
    return x


# 转换张量的类型
def astype_tensor(tensor, dtype):
    """
    使用.fn()的方式直接转化tensor的数据类型
    """
    if dtype == 'float':
        x = tensor.float()
    elif dtype == 'long':
        x = tensor.long()
    else:
        x = tensor.bool()

    return x


# 随机生成一个dim维的张量
def multidim_tensor(dim):
    """
    常见的张量维度有0 ~ 4
    0维表示标量, 1维表示数组, 2维表示矩阵, 3维表示彩色图像, 4维表示包含batch的图片数据
    """
    x = None
    if dim == 0:
        x = torch.rand(1).item()
    elif dim == 1:
        x = torch.rand(5)
    elif dim == 2:
        x = torch.rand(5, 5)
    elif dim == 3:
        x = torch.rand(3, 5, 5)
    elif dim == 4:
        x = torch.rand(32, 3, 5, 5)

    print(x.dim())
    return x


# 输出张量的尺寸(形状)
def tensor_size(tensor):
    """
    使用tensor.size()或者tensor.shape方法都能输出张量的尺寸
    """
    print(tensor.size())
    print(tensor.shape)
    return None


# 修改张量的尺寸
def resize_tensor(tensor):
    """
    以4维图片数据为例, 将其batch维度保留, 其余维度进行铺展
    """
    batch = tensor.size(0)
    x = tensor.view(batch, -1)
    return x


# torch.Tensor与numpy.Array的相互转换
def tensor2array(tensor):
    """
    tensor转为array
    """
    x = tensor.numpy()
    print(type(x))
    return x


def array2tensor(array):
    """
    array转为tensor
    """
    x = torch.from_numpy(array)
    print(type(x))
    return x


# 获取张量的具体数值
def get_tensor(tensor):
    """
    当tensor的维度等于1时, 可获取其值, 否则获取其列表形式
    """
    size = tensor.dim()
    if size == 1:
        x = tensor.item()
    else:
        x = tensor.tolist()

    return x


if __name__ == "__main__":
    # 学习张量的数据类型
    t1 = create_tensor(2, torch.long)
    print(t1)
    t2 = astype_tensor(t1, 'float')
    print(t2)
    # 学习生成多维度张量
    t3 = multidim_tensor(2)
    print(t3)
    t4 = multidim_tensor(4)
    print(t4)
    # 打印张量的维度
    tensor_size(t3)
    tensor_size(t4)
    # 修改张量的尺寸
    t5 = resize_tensor(t4)
    print(t5)
    # 将张量转为numpy数组
    arr1 = tensor2array(t5)
    print(arr1)
    # 将numpy数组转为张量
    arr2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    t6 = array2tensor(arr2)
    print(t6.dim())
    # 获取tensor的数值
    val1 = get_tensor(t1)
    print(val1)
    val2 = get_tensor(t6)
    print(val2)
