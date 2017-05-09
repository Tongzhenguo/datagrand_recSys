# coding=utf-8
import pandas as pd
import numpy as np
"""

"""

def Recommendation_s(uid, k=5, iid_iid_sim=rate_cos_s):
    predict = pd.DataFrame()
    uid_action = rate_mat.loc[uid, :].values  # 用户uid 对所有商品的行为评分
    iid_sim = iid_iid_sim.loc[iid, :].values  # 商品iid 对所有商品的相似度

    for j in uid_action.index:  # 评价物品的index:
        item_id = uid_action
        score = uid_action[j] * iid_iid_sim.loc[j]


