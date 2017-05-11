# coding=utf-8
from itertools import permutations
import math
import pandas as pd
import numpy as np
"""
    去掉已经view的
    实现loglike相似度
"""

def get_rating_matrix(  ):
    train = pd.read_csv('../data/train.csv')
    news = pd.read_csv('../data/news_info.csv')
    item_view = pd.read_csv('../data/filterItems.csv')

    # 选择距end时间2小时内被view过的，其余的训练集item假定已经失去了时效，不再推荐
    end = train['action_time'].max()
    df = train[train.action_time >= end - 2 * 60 * 60][['user_id', 'item_id']]
    item_view['action_type'] = ['view'] * len(item_view)
    df = df.append(item_view)

    train = pd.merge(df, news, on='item_id')[['user_id','item_id','action_type']]

    train['weight'] = train['action_type'].apply(lambda p:1 if p in ['view' 'deep_view'] else 5)
    train = train[['user_id','item_id','weight']].groupby( ['user_id','item_id'],as_index=False ).sum()
    return train

def get_concur_mat(  ):
    rat_mat = get_rating_matrix()
    sim_mat = pd.DataFrame()
    item1_list = []
    item2_list = []
    concur_count = []
    user_groups = rat_mat.groupby( ['user_id'] )
    for name,group in user_groups:
        for pair in permutations(list(group['item_id'].values), 2):
            item1_list.append( pair[0] )
            item2_list.append( pair[1] )
            concur_count.append( 1 )
        # print name
    sim_mat['item1'] = item1_list
    sim_mat['item2'] = item2_list
    sim_mat['count'] = concur_count
    sim_mat = sim_mat.groupby(['item1', 'item2'], as_index=False).sum()
    return sim_mat

def get_concur_sim(  ):
    concur_mat = get_concur_mat()
    rat_mat = get_rating_matrix()
    item_vector = rat_mat[['item_id','user_id']].groupby(['item_id'],as_index=False).count()
    item_vector.index = item_vector['item_id']
    item_vector.columns = ['item_id','count']
    item_count_dict = item_vector['count'].to_dict()
    concur_mat['item1_count'] = concur_mat['item1'].apply( lambda p:item_count_dict[p] )
    concur_mat['item2_count'] = concur_mat['item2'].apply(lambda p: item_count_dict[p])
    concur_mat['sim'] = concur_mat['count'] / (concur_mat['item1_count'].apply(math.sqrt) * concur_mat['item2_count'].apply(math.sqrt))

    sim_mat = pd.DataFrame()
    for item1,group in concur_mat.groupby( ['item1'],as_index=False ):
        df = group.sort_values( ['sim'],ascending=False ).head( 10 )
        df['item1'] = [item1] * len(df)
        # print('------------------------------')
        sim_mat = sim_mat.append( df )
    return sim_mat[['item1','item2','sim']]

def help( p ):
    ss = str(p).split(",")
    rec,viewed = ss[0],ss[1]
    rec = list( rec.split(" ") )

    size = 0
    for i in rec:
        size += 1
        if i in list( set( viewed.split(" ") ) ):
            rec.remove( i )
            size -= 1
        if size == 5:break

    rec = " ".join(rec[:5])
    return rec

def Recommendation_s( k=5):
    print('计算评分矩阵')
    rate_mat = get_rating_matrix()
    print('计算共现矩阵')
    iid_iid_sim = get_concur_sim()
    item_view = pd.read_csv('../data/filterItems.csv')

    rec = pd.DataFrame()
    user_list = []
    rec_items_list = []
    users = pd.read_csv('../data/candidate.txt')
    df = pd.merge( users,rate_mat,on='user_id' )
    df = pd.merge( df,iid_iid_sim,left_on='item_id',right_on='item1' )
    df['score'] = df['weight']*df['sim']
    # df = pd.merge( users,df,on='user_id' )
    df = df[['user_id','item2','score']].sort_values(['user_id','score'],ascending=False)
    print('为每个用户推荐')
    for user_id,group in df.groupby( ['user_id'],as_index=False,sort=False ):
        rec_items = " ".join(map(str, list(group['item2'].head(10).values)))

        user_list.append(user_id)
        rec_items_list.append( rec_items )
        # print('------------------------')
    rec['user_id'] = user_list
    rec['item_id'] = rec_items_list
    print('过滤掉用户已经看过的')
    rec = pd.merge(rec, item_view, how='left', on='user_id').fillna("")
    rec['item_id'] = rec['item_id_x'] + "," + rec['item_id_y']
    rec['item_id'] = rec['item_id'].apply(help)
    print('还有1w的冷启动用户,推荐全部候选用户所关注的全局最新最热的')
    oldrec = pd.read_csv('../result/result_063837_filter.csv')
    oldrec = pd.merge(oldrec, rec, how='left', on='user_id', ).fillna(0)
    oldrec = oldrec[oldrec.rec_item == 0][['user_id', 'oldrec_item']]
    oldrec.columns = ['user_id', 'rec_item']
    rec = rec.append(oldrec)
    rec.drop_duplicates('user_id').to_csv('../result/result', index=None, header=None)

Recommendation_s()
