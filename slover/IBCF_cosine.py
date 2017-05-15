# coding=utf-8
from itertools import permutations
import math
import pandas as pd
import numpy as np
import time
import pickle as pickle
import os
"""
    去掉已经view的
"""

#w(x) = log( 1+N('view') / N(x) ),实际评分和w(x)=1一样
def get_action_weight( x):
    if x == 'view': return 1
    if x == 'deep_view': return 2
    if x == 'share':return 8
    if x == 'comment': return 6
    if x == 'collect':return 5
    else:return 1

def get_rating_matrix(  ):
    path = '../cache/rating_weight.pkl'
    if os.path.exists(path):
        train = pickle.load(open(path, "rb"))
    else:
        start = time.mktime(time.strptime('2017-2-18 18:00:00', '%Y-%m-%d %H:%M:%S')) #测试最近6个小时的，线上分数最高
        train = pd.read_csv('../data/train.csv')
        train = train[ (train['action_time']<start) ][['user_id', 'item_id', 'action_type']]

        item_display = pd.read_csv('../data/item_display.csv')
        item_display['end_time'] = item_display['end_time'].apply( lambda x: time.mktime(time.strptime(x, '%Y%m%d %H:%M:%S')) )
        # 选择距end时间2小时内被view过的，其余的训练集item假定已经失去了时效，不再推荐
        end = time.mktime(time.strptime('2017-2-18 22:00:00', '%Y-%m-%d %H:%M:%S'))
        news = item_display[(item_display['end_time'] >= end)][['item_id']]
        train = pd.merge(train, news, on='item_id')[['user_id','item_id','action_type']]

        train['weight'] = train['action_type'].apply( get_action_weight )
        train = train[['user_id','item_id','weight']].groupby( ['user_id','item_id'],as_index=False ).sum()
        train.to_csv('../data/rat_mat_test.csv',index=False)
        pickle.dump( train,open(path,'wb'),True ) #dump 时如果指定了 protocol 为 True，压缩过后的文件的大小只有原来的文件的 30%
    return train

#可以优化空间，存储成三角矩阵
def get_concur_mat(  ):
    path = "../cache/get_concur_mat.pkl"
    if os.path.exists(path):
        sim_mat = pickle.load(open(path, "rb"))
    else:
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
        pickle.dump(sim_mat, open(path, 'wb'), True)  # dump 时如果指定了 protocol 为 True，压缩过后的文件的大小只有原来的文件的 30%
    return sim_mat

def get_concur_sim(  ):
    path = "../cache/cosine_sim_mat.pkl"
    if os.path.exists(path):
        sim_mat = pickle.load(open(path, "rb"))
    else:
        concur_mat = get_concur_mat()
        print('----------------load concur_mat--------------------')
        rat_mat = get_rating_matrix()
        print('----------------load rat_mat--------------------')
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
            # print('---------------------------')
        sim_mat = sim_mat[['item1', 'item2', 'sim']]
        pickle.dump(sim_mat, open(path, 'wb'), True)
    return sim_mat

def help( p ):
    ss = str(p).split(",")
    rec,viewed = ss[0],ss[1]
    rec = list( rec.split(" ") )
    viewed_list = list( set( viewed.split(" ") ) )
    size = 0
    for i in rec:
        size += 1
        if i in viewed_list:
            rec.remove( i )
            size -= 1
        if size == 5:break

    rec = " ".join(rec[:5])
    return rec

def Recommendation():
    train = pd.read_csv('../data/train.csv')
    train['item_id'] = train['item_id'].apply(str)
    print('计算评分矩阵')
    rate_mat = get_rating_matrix()
    rate_mat['item_id'] = rate_mat['item_id'].apply(str)
    print('计算相似度')
    iid_iid_sim = get_concur_sim()
    iid_iid_sim['item1'] = iid_iid_sim['item1'].apply(str)
    iid_iid_sim['item2'] = iid_iid_sim['item2'].apply(str)
    user_list = []
    viewed_list = []
    for user,group in train[['user_id','item_id']].groupby(['user_id']):
        user_list.append( user )
        viewed_list.append( " ".join( list(group['item_id'].values) ) )
    user_viewed = pd.DataFrame()
    user_viewed['user_id'] = user_list
    user_viewed['item_id'] = viewed_list

    rec = pd.DataFrame()
    user_list = []
    rec_items_list = []
    users = pd.read_csv('../data/candidate.txt')
    df = pd.merge( users,rate_mat,on='user_id' )
    df = pd.merge( df,iid_iid_sim,left_on='item_id',right_on='item1' )
    df['score'] = df['weight']*df['sim']
    df = df[['user_id','item2','score']].sort_values(['user_id','score'],ascending=False)
    print('为每个用户推荐')
    for user_id,group in df.groupby( ['user_id'],as_index=False,sort=False ):
        rec_items = " ".join(map(str, list(group['item2'].head(15).values)))

        user_list.append(user_id)
        rec_items_list.append( rec_items )
        # print('------------------------')
    rec['user_id'] = user_list
    rec['item_id'] = rec_items_list

    print('还有部分的冷启动用户,推荐18时之后的topHot20') #这里也可改进
    train_h18 = train[train.action_time >= time.mktime(time.strptime('2017-2-18 18:00:00', '%Y-%m-%d %H:%M:%S'))]
    topHot = train_h18.groupby(['item_id'], as_index=False).count().sort_values(['action_time'], ascending=False).head(15)[
        'item_id'].values
    oldrec = users
    oldrec['oldrec_item'] = [" ".join(list(topHot))] * len(oldrec)
    oldrec = pd.merge(oldrec, rec, how='left', on='user_id', ).fillna(0)
    oldrec = oldrec[oldrec.item_id == 0][['user_id', 'oldrec_item']]
    oldrec.columns = ['user_id', 'item_id']
    rec = rec.append(oldrec)

    print('过滤掉用户已经看过的')
    rec = pd.merge(rec,user_viewed , how='left', on='user_id').fillna("") #item_view
    rec['item_id'] = rec['item_id_x'] + "," + rec['item_id_y']
    rec['item_id'] = rec['item_id'].apply(help)
    rec = rec[['user_id','item_id']]

    rec.drop_duplicates('user_id').to_csv('../result/result.csv', index=None, header=None) #0.009568

if __name__ == "__main__":
    get_rating_matrix()
