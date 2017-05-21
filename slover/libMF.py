# coding=utf-8
import csv
import os

import gc

import numpy
import pandas as pd

#w(x) = log( 1+N('view') / N(x) ),实际评分和w(x)=1一样
import time


def get_action_weight( x):
    if x == 'view': return 1
    if x == 'deep_view': return 2
    if x == 'share':return 8
    if x == 'comment': return 6
    if x == 'collect':return 5
    else:return 1

def make_train_test(  ):
    train = pd.read_csv('../data/train.csv')
    user = pd.read_csv('../data/candidate.txt')
    item = pd.read_csv('../data/all_news_info.csv')

    #将user_id和item_id重新映射成连续的id
    uid_uniqid = user[['user_id']].sort_values(['user_id'])
    uid_uniqid.index = user['user_id'].values
    uid_uniqid['user_id'] = range(len(uid_uniqid))
    uid_uniqid = uid_uniqid['user_id'].to_dict()

    iid_uniqid = item[['item_id']].sort_values(['item_id'])
    iid_uniqid.index = item['item_id'].values
    iid_uniqid['item_id'] = range(len(iid_uniqid))
    iid_uniqid = iid_uniqid['item_id'].to_dict()

    train['weight'] = train['action_type'].apply(get_action_weight)
    train = pd.merge(user, train, on='user_id')
    rat_mat = train[['user_id', 'item_id', 'weight']].groupby(['user_id', 'item_id'], as_index=False).sum()
    rat_mat['user_id'] = rat_mat['user_id'].apply( lambda x: uid_uniqid.get(x) )
    rat_mat['item_id'] = rat_mat['item_id'].apply( lambda x: iid_uniqid.get(x) )
    rat_mat['weight'] = rat_mat['weight'].apply(float)
    rat_mat.to_csv('../data/real_matrix.tr.txt', index=False, header=False,sep=" ")

def save_user_mat( factor_num, ):
    f = open('../model/libMF_model_l1l2', 'r')
    user_mat_csv = open('../data/user_mat.csv', 'w')
    csv_writer = csv.writer(user_mat_csv,delimiter=',')
    csv_writer.writerow( ['uniqid', 'flag']+["factor_" + str(i) for i in range(factor_num)] )
    n = 0
    while( True ):
        line = f.readline()
        if(line.startswith("p")):
            #去掉p标志
            ss = line[1:].strip().split(" ")
            csv_writer.writerow( ss )
            n += 1
        if( n % 1000 == 0 ): print("write lines "+str(n))
        if ( line == None or line.startswith("q") ):
            break
    f.close()
    user_mat_csv.close()
    print(' write all lines '+str(n) )

def save_item_mat( factor_num, ):
    f = open('../model/libMF_model_l1l2', 'r')
    item_mat_csv = open('../data/item_mat.csv', 'w')
    csv_writer = csv.writer(item_mat_csv,delimiter=',')
    csv_writer.writerow( ['uniqid', 'flag']+["factor_" + str(i) for i in range(factor_num)] )
    n = 0
    while( True ):
        line = f.readline()
        if(line.startswith("q")):
            #去掉标志
            ss = line[1:].strip().split(" ")
            csv_writer.writerow( ss )
            n += 1
        if( n % 1000 == 0 ): print("write lines "+str(n))
        if ( not line ):
            break
    f.close()
    item_mat_csv.close()
    print(' write all lines '+str(n) )

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

    rec = " ".join( rec[:5])
    return rec

def addAndSortTopK( e,sorted_list,k=60 ):
    if( len(sorted_list)<k ):
        sorted_list.append( e )
    if( len(sorted_list)>=k and e[1]>sorted_list[k-1][1] ):
        sorted_list.append( e )
        sorted_list.sort(key=lambda x:-x[1])
    return sorted_list


def make_predict( num_factor ):
    print(' 读取用户和物品矩阵 ')
    user_mat = pd.read_csv('../data/user_mat.csv')
    item_mat = pd.read_csv('../data/item_mat.csv')
    item = pd.read_csv('../data/news_info.csv')
    train = pd.read_csv('../data/train.csv')
    user = pd.read_csv('../data/candidate.txt')
    item_all = pd.read_csv('../data/all_news_info.csv')

    print(' 将uniqid重新映射成user_id,item_id ')
    uniqid_uid = user[['user_id']].sort_values(['user_id'])
    uniqid_uid.index = range(len(uniqid_uid))
    uniqid_uid = uniqid_uid['user_id'].to_dict()

    uniqid_iid = item_all[['item_id']].sort_values(['item_id'])
    uniqid_iid.index = range(len(uniqid_iid))
    uniqid_iid = uniqid_iid['item_id'].to_dict()

    user_mat['user_id'] = user_mat['uniqid'].apply( lambda x:uniqid_uid[x] )
    item_mat['item_id'] = item_mat['uniqid'].apply( lambda x: uniqid_iid[x] )

    #这里有些新品是没处理的，可以通过同cate的隐向量进行均值填充
    item_mat = pd.merge( item[['item_id']],item_mat,on='item_id' )
    print(' 去掉空值减少计算量 ')
    item_mat = item_mat[ item_mat['flag']=='T']
    item_mat.index = range(len(item_mat))
    print(' 待推荐item总数'+str(len(item_mat)))

    print(' 过滤掉那些阅读晚高峰也没被看过的item,大约1w多 ')
    start = time.mktime(time.strptime('2017-2-18 18:00:00', '%Y-%m-%d %H:%M:%S'))
    item_max_time = train.groupby(['item_id'], as_index=False).max()[['item_id', 'action_time']]
    item_max_time = item_max_time[item_max_time['action_time'] > start]
    item_mat = pd.merge( item_max_time[['item_id']],item_mat,on='item_id' )

    print(' 测试集里top10的item_id ')
    test = pd.read_csv('../data/test.csv')
    test = test.groupby(['item_id'],as_index=False).count().sort_values(['user_id'],ascending=False)[:10]
    item_mat = pd.merge(item_mat, test[['item_id']].drop_duplicates(), on='item_id')

    print( ' 预测评分 ' )
    rec = pd.DataFrame()
    user_list = []
    rec_items_list = []
    sorted_list = []
    n = 0
    feat = ["factor_" + str(i) for i in range(num_factor)]
    user_mat = user_mat[ ['user_id']+feat ]
    item_mat = item_mat[ ['item_id']+feat ]
    for i in range( len(user_mat) ):
        recitems = []
        for j in range( len(item_mat) ):
            predict = user_mat.ix[i,1:].dot( item_mat.ix[j,1:] )
            addAndSortTopK( [item_mat.ix[j,0],predict],sorted_list )
        for item_predict in sorted_list:
            recitems.append( int(item_predict[0]) )
        sorted_list.clear()
        user_list.append( user_mat.ix[i,0] )
        rec_items_list.append( " ".join( map(str,recitems) ) )
        n += 1
        if( n%2==0 ):print(' rec users '+str( n ))
    rec['user_id'] = user_list
    rec['item_id'] = rec_items_list
    del item_all
    del user
    del item
    del user_list
    del rec_items_list
    gc.collect()

    print('过滤掉用户已经看过的')
    user_viewed = pd.DataFrame()
    user_list = []
    viewed_item = []
    for user, group in train[['user_id', 'item_id']].groupby(['user_id'], as_index=False):
        user_list.append(user)
        viewed_item.append(" ".join(  map( str, map( int,list(group['item_id'].unique())) )))
    user_viewed['user_id'] = user_list
    user_viewed['item_id'] = viewed_item
    del user_list
    del viewed_item
    gc.collect()
    rec = pd.merge(rec, user_viewed, how='left', on='user_id').fillna("")
    rec['item_id'] = rec['item_id_x'] + "," + rec['item_id_y']
    rec['item_id'] = rec['item_id'].apply(help)
    rec = rec[['user_id', 'item_id']]
    rec.drop_duplicates('user_id').to_csv('../result/result.csv', index=None, header=None)


if __name__ == '__main__':
    # make_train_test()
    # exit_num = os.system("../bins/libMF.sh")
    # print(  exit_num >> 8 )
    # save_user_mat(35)
    # save_item_mat(35)
    make_predict(35)