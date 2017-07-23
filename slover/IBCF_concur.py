# coding=utf-8
import os
from itertools import permutations
import math
import pickle as pickle
import pandas as pd
import time

"""
    基于物品的协同过滤算法，相似度是共现矩阵：
    sim = iid1_iid2_cnt / sqrt( iid1 * iid2 )
    去掉已经view的
    有趣的是，全量计算（16日起）并没有只计算最近6小时（18日18时起）的物品相似度效果好，这个是之前测试的

"""

def get_rating_matrix( train,test ):
    path = '../cache/rating_all_test.pkl'
    if os.path.exists(path):
        train = pickle.load(open(path, "rb"))
    else:
        #明显，在test中出现的是还有时效的，所以只要test和train做交集就可以了
        train = pd.merge( train,test[['item_id']],on='item_id' )
        train = train[train.action_type=='deep_view'][['user_id','item_id']].drop_duplicates()
        # print len(train),train['item_id'].unique().shape[0]
        pickle.dump(train, open(path, 'wb'), True)  # dump 时如果指定了 protocol 为 True，压缩过后的文件的大小只有原来的文件的 30%
    return train

def get_concur_mat( train ):
    path = "../cache/concur_mat_test.pkl"
    if os.path.exists(path):
        sim_mat = pickle.load(open(path, "rb"))
    else:
        rat_mat = get_rating_matrix( train )
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
        pickle.dump(sim_mat, open(path, 'wb'), True)
    return sim_mat

def get_concur_sim( train,test ):
    path = "../cache/concur_sim_mat_test.pkl"
    if os.path.exists(path):
        sim_mat = pickle.load(open(path, "rb"))
    else:
        concur_mat = get_concur_mat( train )
        rat_mat = get_rating_matrix( train,test )
        item_vector = rat_mat[['item_id','user_id']].groupby(['item_id'],as_index=False).count()
        item_vector.index = item_vector['item_id']
        item_vector.columns = ['item_id','count']
        item_count_dict = item_vector['count'].to_dict()
        concur_mat['item1_count'] = concur_mat['item1'].apply( lambda p:item_count_dict[p] )
        concur_mat['item2_count'] = concur_mat['item2'].apply(lambda p: item_count_dict[p])
        concur_mat['sim'] = concur_mat['count'] / (concur_mat['item1_count'].apply(math.sqrt) * concur_mat['item2_count'].apply(math.sqrt))

        sim_mat = pd.DataFrame()
        for item1,group in concur_mat.groupby( ['item1'],as_index=False ):
            df = group.sort_values( ['sim'],ascending=False ).head(20)
            sim_mat = sim_mat.append( df )
        pickle.dump(sim_mat, open(path, 'wb'), True)
        sim_mat[['item1', 'item2', 'sim']].to_csv('../data/item_ralation.csv')
    return sim_mat[['item1','item2','sim']]

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

def Recommendation(k=10):
    test = pd.read_csv('../data/test.csv')
    train = pd.read_csv('../data/train.csv')
    users = pd.read_csv('../data/candidate.txt')
    train = pd.merge( test,train,on='item_id' )
    train['item_id'] = train['item_id'].apply(str)
    print('计算评分矩阵')
    rate_mat = get_rating_matrix( train,test )
    rate_mat['item_id'] = rate_mat['item_id'].apply(str)
    print('计算相似度')
    iid_iid_sim = get_concur_sim( train,test )
    iid_iid_sim['item1'] = iid_iid_sim['item1'].apply(str)
    iid_iid_sim['item2'] = iid_iid_sim['item2'].apply(str)
    user_list = []
    viewed_list = []
    for user, group in train[['user_id', 'item_id']].groupby(['user_id']):
        user_list.append(user)
        viewed_list.append(" ".join(list(group['item_id'].values)))
    user_viewed = pd.DataFrame()
    user_viewed['user_id'] = user_list
    user_viewed['item_id'] = viewed_list

    rec = pd.DataFrame()
    user_list = []
    rec_items_list = []
    df = pd.merge(users, rate_mat, on='user_id')
    df = pd.merge(df, iid_iid_sim, left_on='item_id', right_on='item1')
    df['score'] = df['weight'] * df['sim']
    df = df[['user_id', 'item2', 'score']].sort_values(['user_id', 'score'], ascending=False)
    print('为每个用户推荐')
    for user_id, group in df.groupby(['user_id'], as_index=False, sort=False):
        rec_items = " ".join(map(str, list(group['item2'].values)))

        user_list.append(user_id)
        rec_items_list.append(rec_items)
    rec['user_id'] = user_list
    rec['item_id'] = rec_items_list

    print('过滤掉用户已经看过的')
    rec = pd.merge(rec, user_viewed, how='left', on='user_id').fillna("")  # item_view
    rec['item_id'] = rec['item_id_x'] + "," + rec['item_id_y']
    rec['item_id'] = rec['item_id'].apply(help)
    rec = rec[['user_id', 'item_id']]

    print('还有部分用户没关注过物品候选集,推荐test中的topHot5')
    topHot = \
        test.groupby(['item_id'], as_index=False).count().sort_values(['user_id'], ascending=False).head(5)[
        'item_id'].values
    oldrec = users
    oldrec['oldrec_item'] = [" ".join( map( str,list(topHot) ) )] * len(oldrec)
    oldrec = pd.merge(oldrec, rec, how='left', on='user_id', ).fillna(0)
    oldrec = oldrec[oldrec.item_id == 0][['user_id', 'oldrec_item']]
    oldrec.columns = ['user_id', 'item_id']
    rec = rec.append(oldrec)

    rec.drop_duplicates('user_id').to_csv('../result/result_cf.csv', index=None, header=None)

if __name__ == "__main__":
    Recommendation()
