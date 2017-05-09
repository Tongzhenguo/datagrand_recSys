# coding=utf-8
import pandas as pd
import numpy as np
"""

"""

def Recommendation_s( rate_mat,iid_iid_sim,k=5):
    rec = pd.DataFrame()
    user_list = []
    rec_items_list = []
    iid_iid_sim = iid_iid_sim.sort_values(['item1','sim'],ascending=False)
    # users = list(pd.read_csv('../../data/candidate.txt')['user_id'].values)

    df = pd.merge( rate_mat,iid_iid_sim,left_on='item_id',right_on='item1' )
    df['score'] = df['weight']*df['sim']
    df = df[['user_id','item2','score']].sort_values(['user_id','score'],ascending=False)
    for u in list(df['user_id'].unique()):
        rec_items = " ".join( map( str,list(df[ df['user_id']==u ]['item2'].head(k).values)) )
        user_list.append( u )
        rec_items_list.append( rec_items )
    rec['user_id'] = user_list
    rec['rec_items'] = rec_items_list

    # 还有1w的冷启动用户,直接推荐全局最热的
    oldrec = pd.read_csv('../result/result_0.007403.csv')
    oldrec = pd.merge(oldrec, rec, how='left', on='user_id', ).fillna(0)
    oldrec = oldrec[oldrec.rec_item == 0][['user_id', 'oldrec_item']]
    oldrec.columns = ['user_id', 'rec_item']
    rec = rec.append(oldrec)
    rec.to_csv('../result/result.csv', index=None, header=None)

rate_mat = pd.read_csv('../../data/rating.csv')
iid_iid_sim = pd.read_csv('../../data/sim/concurrence_sim.csv')
Recommendation_s( rate_mat,iid_iid_sim,5 )
