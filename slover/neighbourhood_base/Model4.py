# coding=utf-8
import pandas as pd
import numpy as np
"""

"""
test = pd.read_csv('data/test.csv')
test_arr = test.values.copy()

rate_mat = pd.read_csv('data/rate_mat.csv', index_col=0)  # 用户-物品评分矩阵
rate_cos = pd.read_csv('data/rate_cos.csv', index_col=0)  # 基于余弦相似度的物品评分矩阵
rate_cos_s = pd.read_csv('data/rate_cos_s.csv', index_col=0)  # 基于改进余弦相似度的物品评分矩阵
rate_pearson = pd.read_csv('data/rate_pearson.csv', index_col=0)  # 基于皮尔逊相关系数的物品评分矩阵

rate_mat = rate_mat.rename(columns=dict(zip(rate_mat.columns, [int(i) for i in rate_mat.columns])))
rate_cos = rate_cos.rename(columns=dict(zip(rate_cos.columns, [int(i) for i in rate_cos.columns])))
rate_cos_s = rate_cos_s.rename(columns=dict(zip(rate_cos_s.columns, [int(i) for i in rate_cos_s.columns])))
rate_pearson = rate_pearson.rename(columns=dict(zip(rate_pearson.columns, [int(i) for i in rate_pearson.columns])))

iid_index = rate_mat.columns


def Recommendation_s(uid, iid, k=10, iid_iid_sim=rate_cos_s):
    score = 0
    weight = 0
    iid_sim = iid_iid_sim.loc[iid, :].values  # 商品iid 对所有商品的相似度
    uid_action = rate_mat.loc[uid, :].values  # 用户uid 对所有商品的行为评分
    iid_action = rate_mat.loc[:, iid].values  # 物品iid 得到的所有用户评分
    sim_indexs = np.argsort(iid_sim)[-(k + 1):-1]  # 最相似的k个物品的index

    iid_i_mean = np.sum(iid_action) / iid_action[iid_action != 0].size
    for j in sim_indexs:
        if uid_action[j] != 0:
            iid_j_action = rate_mat.values[:, j]
            iid_j_mean = np.sum(iid_j_action) / iid_j_action[iid_j_action != 0].size
            score += iid_sim[j] * (uid_action[j] - iid_j_mean)
            weight += abs(iid_sim[j])

    if weight == 0:
        return iid_i_mean  # 可以再改进！
    else:
        return iid_i_mean + score / float(weight)


def Recommendation(uid, iid, k=10, iid_iid_sim=rate_cos_s):
    score = 0
    weight = 0
    iid_sim = iid_iid_sim.loc[iid, :].values  # 商品iid 对所有商品的相似度
    uid_action = rate_mat.loc[uid, :].values  # 用户uid 对所有商品的行为评分
    iid_action = rate_mat.loc[:, iid].values  # 物品iid 得到的所有用户评分
    sim_indexs = np.argsort(iid_sim)[-(k + 1):-1]  # 最相似的k个物品的index

    iid_mean = np.sum(iid_action) / iid_action[iid_action != 0].size
    for j in sim_indexs:
        if uid_action[j] != 0:
            score += iid_sim[j] * (uid_action[j] - iid_mean)
            weight += abs(iid_sim[j])

    if weight == 0:
        return iid_mean  # 可以再改进！
    else:
        return iid_mean + score / float(weight)


Num = len(test)
result = np.zeros(Num)


def pred(k, iid_iid_sim):
    count = 0
    for i in range(Num):
        a = test_arr[i, 0]
        b = test_arr[i, 1]
        if b not in iid_index:
            result[i] = 3
            count = count + 1
        else:
            result[i] = Recommendation_s(a, b, k, iid_iid_sim)
    print 'count:', count