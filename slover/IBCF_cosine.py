# coding=utf-8
from itertools import permutations
import math
import pandas as pd
import numpy as np
"""
    去掉已经view的
"""

def get_action_weight( x):
    if x == 'view': return 1
    if x == 'deep_view': return 5
    if x == 'share':return 10
    if x == 'comment': return 5
    if x == 'collect':return 15
    else:return 1
def get_rating_matrix(  ):
    train = pd.read_csv('../data/train.csv')
    news = pd.read_csv('../data/news_info.csv')
    # item_view = pd.read_csv('../data/filterItems.csv')

    # 选择距end时间2小时内被view过的，其余的训练集item假定已经失去了时效，不再推荐
    # end = train['action_time'].max()
    df = train[['user_id', 'item_id','action_type']] #2017/2/18 22:0:0 [train.action_time >= 1487426400]
    # item_view['action_type'] = ['view'] * len(item_view)
    # df = df.append(item_view)

    train = pd.merge(df, news, on='item_id')[['user_id','item_id','action_type']]

    train['weight'] = train['action_type'].apply( get_action_weight )
    train = train[['user_id','item_id','weight']].groupby( ['user_id','item_id'],as_index=False ).sum()
    train.to_csv('../data/rating_weight.csv',index=None)
    return train

def get_concur_mat(  ):
    rat_mat = get_rating_matrix()
    sim_mat = pd.DataFrame()
    item1_list = []
    item2_list = []
    item1_item2_score = []
    user_groups = rat_mat.groupby( ['user_id'] )
    for name,group in user_groups:
        for pair in permutations(list(group[['item_id','weight']].values), 2):
            item1_list.append( pair[0][0] )
            item2_list.append( pair[1][0] )
            item1_item2_score.append( pair[0][1]*pair[1][1] )
        # print name
    sim_mat['item1'] = item1_list
    sim_mat['item2'] = item2_list
    sim_mat['score'] = item1_item2_score
    sim_mat = sim_mat.groupby(['item1', 'item2'], as_index=False).sum()
    return sim_mat

def get_concur_sim(  ):
    concur_mat = get_concur_mat()
    rat_mat = get_rating_matrix()
    rat_mat['score2'] = rat_mat[['weight']] *  rat_mat[['weight']]
    item_sum_s2_vector = rat_mat[['item_id','score2']].groupby(['item_id'],as_index=False).sum()
    item_sum_s2_vector.index = item_sum_s2_vector['item_id']
    item_sum_s2_dict = item_sum_s2_vector['score2'].to_dict()
    concur_mat['item1_sum_s2'] = concur_mat['item1'].apply( lambda p:item_sum_s2_dict[p] )
    concur_mat['item2_sum_s2'] = concur_mat['item2'].apply(lambda p: item_sum_s2_dict[p])
    concur_mat['sim'] = concur_mat['score'] / (concur_mat['item1_sum_s2'].apply(math.sqrt) * concur_mat['item2_sum_s2'].apply(math.sqrt))
    print('------------      取前20个最相似的item    ------------------')
    sim_mat = pd.DataFrame()
    for item1,group in concur_mat.groupby( ['item1'],as_index=False ):
        df = group.sort_values( ['sim'],ascending=False ).head( 20 )
        df['item1'] = [item1] * len(df)
        sim_mat = sim_mat.append( df )
    sim_mat[['item1', 'item2', 'sim']].to_csv('../data/cosine_sim_mat.csv',index=False)
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
    # rate_mat = pd.read_csv('../data/rating_weight.csv')
    print('计算cosine相似度')
    iid_iid_sim = get_concur_sim()
    # iid_iid_sim = pd.read_csv('../data/concur_sim_mat.csv')
    # item_view = pd.read_csv('../data/filterItems.csv')

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
        rec_items = " ".join(map(str, list(group['item2'].head(15).values)))

        user_list.append(user_id)
        rec_items_list.append( rec_items )
        # print('------------------------')
    rec['user_id'] = user_list
    rec['item_id'] = rec_items_list
    print('过滤掉用户已经看过的')
    rec = pd.merge(rec, item_view, how='left', on='user_id').fillna("")
    rec['item_id'] = rec['item_id_x'] + "," + rec['item_id_y'].apply(str)
    rec['item_id'] = rec['item_id'].apply(help)
    rec = rec[['user_id','item_id']]

    print('过滤掉')

    print('还有1w的冷启动用户,推荐全部候选用户所关注的全局最新最热的')
    oldrec = pd.read_csv('../result/result_063837_filter.csv')
    oldrec = pd.merge(oldrec, rec, how='left', on='user_id', ).fillna(0)
    oldrec = oldrec[oldrec.item_id == 0][['user_id', 'oldrec_item']]
    oldrec.columns = ['user_id','item_id']
    oldrec = oldrec[['user_id','item_id']]
    rec = rec.append(oldrec)
    rec.drop_duplicates('user_id').to_csv('../result/test.csv', index=None, header=None)

Recommendation_s()
# get_rating_matrix()