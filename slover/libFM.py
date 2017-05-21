import os
import random

import gc
import pandas as pd
import pickle

import time
from sklearn.model_selection import train_test_split


def randomSelectTrain( train ):
    path = '../cache/train.pkl'
    if os.path.exists( path ):
        data = pickle.load(open(path, "rb"))
    else:
        # 抽样选择与用户阅读资讯相同数量的热门资讯
        data = pd.DataFrame()
        user_list = []
        item_list = []
        label_list = []
        items_pool = list(train['item_id'].values)
        for user, group in train.groupby(['user_id']):
            rec = dict()
            viewed_item = list(group['item_id'].unique() )
            for item in viewed_item:
                rec[item] = 1
                user_list.append(user)
                item_list.append(item)
                label_list.append(1)
            n = 0
            for i in range( 0, len(items_pool) ):
                item = items_pool[random.randint(0, len(items_pool) - 1)]
                if item in rec.keys():
                    continue
                user_list.append(user)
                item_list.append(item)
                label_list.append(0)
                n += 1
                rec[item] = 0
                if n > len(viewed_item):
                    break
        data['user_id'] = user_list
        data['item_id'] = item_list
        data['label'] = label_list
        del user_list
        del item_list
        del label_list
        gc.collect()
        pickle.dump( data,open(path,'wb'),True )
    return data


def make_train_test(  ):
    train = pd.read_csv('../data/train.csv')
    data = randomSelectTrain( train )
    data.to_csv('../data/trainData.csv', index=False, header=False)

    # train,test = train_test_split(data,test_size=0.2,random_state=199009)
    # train.to_csv( '../data/trainData.csv',index=False,header=False )
    # test.to_csv('../data/testData.csv', index=False, header=False)

def make_predict_test(  ):
    train = pd.read_csv('../data/train.csv')
    user = pd.read_csv('../data/candidate.txt')
    start = time.mktime(time.strptime('2017-2-18 18:00:00', '%Y-%m-%d %H:%M:%S'))
    train = train[ train['action_time']>=start ]
    train = pd.merge( user,train,how='left',on='user_id' )[['user_id','item_id']].fillna(0)
    rec_item = list( train.groupby(['item_id'],as_index=False).count().sort_values( ['user_id'],ascending=False )['item_id'].values )
    predict = pd.DataFrame()
    user_list = []
    rec_list = []
    for user,group in train.groupby( ['user_id'],as_index=False ):
        viewed_item = list( group['item_id'].unique() )
        n = 0
        for item in rec_item:
            if item in viewed_item:
                continue
            n += 1
            user_list.append( user )
            rec_list.append( item )
            if n >= 100:
                break
    predict['user_id'] = user_list
    predict['item_id'] = rec_list
    predict['label'] = 0
    predict.to_csv('../data/testData.csv',index=False,header=False)

def make_result(  ):
    user = pd.read_csv('../data/candidate.txt')
    rec = pd.read_csv('../data/testData.csv')
    score = pd.read_csv('../data/predict.txt')
    rec['score'] = list( score['score'].values )
    res = pd.DataFrame()
    user_list = []
    item_list = []
    for user,group in rec.groupby(['user_id'],as_index=False):
        user_list.append( user )
        items = list(group.sort_values(['score'])['item_id'].values)
        if 0 in items:
            items.remove(0)
        item_list.append( " ".join( map(str, map(int,items[:5]) )) )
    res['user_id'] = user_list
    res['item_id'] = item_list
    del user_list
    del item_list
    gc.collect()
    res.to_csv('../result/result.csv',index=False)
    res = pd.read_csv('../result/result.csv')
    res = pd.merge( user,res,on='user_id' )
    res = res.drop_duplicates('user_id')
    res.to_csv('../result/result.csv',index=False,header=False)

if __name__ == "__main__":
    # make_train_test()
    # make_predict_test()
    # f = os.popen('../bins/libFM.sh')
    # for line in f.readlines():
    #     print(line)
    make_result()