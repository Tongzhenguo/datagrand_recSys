# coding=utf-8
import pandas as pd
import numpy as np
"""
    去掉已经view的
    实现loglike相似度
"""

def Recommendation_s( rate_mat,iid_iid_sim,k=5):
    rec = pd.DataFrame()
    user_list = []
    rec_items_list = []
    users = pd.read_csv('../data/candidate.txt')
    df = pd.merge( rate_mat,iid_iid_sim,left_on='item_id',right_on='item1' )
    df['score'] = df['weight']*df['sim']
    df = pd.merge( users,df,on='user_id' )
    df = df[['user_id','item2','score']].sort_values(['user_id','score'],ascending=False)

    for user_id,group in df.groupby( ['user_id'],as_index=False,sort=False ):
        rec_items = " ".join(map(str, list(group['item2'].head(k).values)))
        user_list.append(user_id)
        rec_items_list.append( rec_items )
        # print('------------------------')
    rec['user_id'] = user_list
    rec['rec_item'] = rec_items_list

    # 还有1w的冷启动用户,直接推荐全局最热的
    oldrec = pd.read_csv('../result/result_0.063837.csv')
    oldrec = pd.merge(oldrec, rec, how='left', on='user_id', ).fillna(0)
    oldrec = oldrec[oldrec.rec_item == 0][['user_id', 'oldrec_item']]
    oldrec.columns = ['user_id', 'rec_item']
    rec = rec.append(oldrec)
    rec.drop_duplicates('user_id').to_csv('../result/result_concur_sim.csv', index=None, header=None)

rate_mat = pd.read_csv('../data/rating.csv')
iid_iid_sim = pd.read_csv('../data/sim/concurrence_sim.csv')
Recommendation_s( rate_mat,iid_iid_sim,5 )
