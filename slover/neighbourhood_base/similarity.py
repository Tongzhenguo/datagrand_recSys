# coding=utf-8
import pandas as pd
import numpy as np

rate_mat = pd.read_csv('../../data/rate_mat.csv').values
def cosine_sim(i, j):
    a = rate_mat[:, i]
    b = rate_mat[:, j]
    m = np.dot(a, b)
    n = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return m / float(n)


def cosine_sim_s(i, j):
    a = rate_mat[:, i]
    b = rate_mat[:, j]
    intersection = a * b
    if intersection[intersection != 0].size == 0:
        return 0.0

    c = a[a != 0]  # 评价物品i的所有用户评分
    d = b[b != 0]
    p = np.mean(c)  # 物品i的所有用户评分均值
    q = np.mean(d)

    m = np.dot(a[intersection != 0] - p, b[intersection != 0] - q)
    n = np.sqrt(np.dot(c - p, c - p) * np.dot(d - q, d - q))
    if n == 0:
        return 0.0
    return m / float(n)


def pearson(i, j):
    a = rate_mat[:, i]
    b = rate_mat[:, j]
    intersection = a * b
    if intersection[intersection != 0].size == 0:
        return 0.0

    c = a[intersection != 0]  # 评价物品i的公共用户评分
    d = b[intersection != 0]
    p = np.mean(a[a != 0])  # 物品i的所有用户评分均值
    q = np.mean(b[b != 0])

    m = np.dot(c - p, d - q)
    n = np.sqrt(np.dot(c - p, c - p) * np.dot(d - q, d - q))
    if n == 0:
        return 0.0
    return m / float(n)


rate_cos = np.zeros((19342, 19342))
rate_cos_s = np.zeros((19342, 19342))
rate_pearson = np.zeros((19342, 19342))

for i in range(19342):
    for j in range(19342):
        if i == j:
            rate_cos[i, j] = 1
        elif rate_cos[j, i] != 0:
            rate_cos[i, j] = rate_cos[j, i]
        else:
            rate_cos[i, j] = cosine_sim(i, j)

for i in range(19342):
    for j in range(19342):
        if i == j:
            rate_cos_s[i, j] = 1
        elif rate_cos_s[j, i] != 0:
            rate_cos_s[i, j] = rate_cos_s[j, i]
        else:
            rate_cos_s[i, j] = cosine_sim_s(i, j)

for i in range(19342):
    for j in range(19342):
        if i == j:
            rate_pearson[i, j] = 1
        elif rate_pearson[j, i] != 0:
            rate_pearson[i, j] = rate_pearson[j, i]
        else:
            rate_pearson[i, j] = pearson(i, j)

iid_index = pd.read_csv('../../data/rate_mat.csv', index_col=0).columns
pd.DataFrame(rate_cos, index=iid_index, columns=iid_index).to_csv('../../data/rate_cos.csv')
pd.DataFrame(rate_cos_s, index=iid_index, columns=iid_index).to_csv('../../data/rate_cos_s.csv')
pd.DataFrame(rate_pearson, index=iid_index, columns=iid_index).to_csv('../../data/rate_pearson.csv')