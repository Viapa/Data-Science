# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: Tensor.py
@date: 2023/9/13 16:05
@target: 学会掌握 Pytorch 中的 Tensor 的相关核心用法
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
    当tensor的维度小于1时, 可获取其值, 否则获取其列表形式
    """
    size = tensor.dim()
    if size < 1:
        x = tensor.item()
    else:
        x = tensor.tolist()

    return x


# einsum函数的应用
def einsum_formula():
    """
    爱因斯坦求和转换规则: C(i,j) = A(i,k)B(k,j)
    1，用元素计算公式来表达张量运算。
    2，只出现在元素计算公式箭头左边的指标叫做哑指标。
    3，省略元素计算公式中对哑指标的求和符号。
    使用einsum(expr, *tensors)函数, 可以输入任意张量运算表达来实现数学运算、矩阵运算和维度变换等功能
    """
    # ------基础用法------
    # 矩阵乘法
    A = torch.tensor([[1, 2], [3, 4.0]])
    B = torch.tensor([[5, 6], [7, 8.0]])
    C1 = A @ B
    print(C1)
    C2 = torch.einsum("ik,kj->ij", A, B)  # 表示将A矩阵(i*k)和B矩阵(k*j)变换为(i*j)形状
    print(C2)
    # 张量转置
    A = torch.randn(3, 4, 5)
    B1 = torch.permute(A, [0, 2, 1])
    print(B1.shape)
    B2 = torch.einsum("ijk->ikj", A)
    print(B2.shape)
    # 取对角线元素
    A = torch.randn(5, 5)
    B1 = torch.diagonal(A)
    print(B1)
    B2 = torch.einsum("ii->i", A)
    print(B2)
    # 降维求和
    A = torch.randn(4, 5)
    B1 = torch.sum(A, dim=1)
    print(B1)
    B2 = torch.einsum("ij->i", A)
    print(B2)
    # 哈达玛积(元素乘法)
    A = torch.randn(5, 5)
    B = torch.randn(5, 5)
    C1 = A * B
    print(C1)
    C2 = torch.einsum("ij,ij->ij", A, B)
    print(C2)
    # 向量内积(点积)
    A = torch.randn(10)
    B = torch.randn(10)
    C1 = torch.dot(A, B)
    print(C1)
    C2 = torch.einsum("i,i->", A, B)
    print(C2)
    # 向量外积(等价于A.T @ B)
    A = torch.randn(10)
    B = torch.randn(5)
    C1 = torch.outer(A, B)
    print(C1.shape)
    C2 = torch.einsum("i,j->ij", A, B)
    print(C2.shape)
    # 张量缩并(广义的矩阵乘法)
    A = torch.randn(3, 4, 5)
    B = torch.randn(4, 3, 6)
    C1 = torch.tensordot(A, B, dims=([0, 1], [1, 0]))
    print(C1.shape)
    C2 = torch.einsum("ijk,jih->kh", A, B)
    print(C2.shape)
    # ------高级用法------
    # 双线性变换



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
    # 学习爱因斯坦求和转换公式应用
    einsum_formula()
