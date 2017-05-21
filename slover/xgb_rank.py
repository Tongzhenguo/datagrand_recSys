# coding=utf-8
import os
import random

import gc
import pandas as pd
import time

import pickle

from sklearn.model_selection import train_test_split
import xgboost as xgb

def randomSelectTrain( train ):
    path = '../cache/train.pkl'
    if os.path.exists( path ):
        data = pickle.load(open(path, "rb"))
    else:
        print('抽样选择与用户阅读资讯相同数量的热门资讯')
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


def show_item_display_time(train, all_news_info):
    item_id_list = []
    start_time_list = []
    end_time_list = []
    df = pd.DataFrame()
    for item_id, group in train.groupby(['item_id'], as_index=False):
        start = group['action_time'].min()
        end = group['action_time'].max()
        item_id_list.append(item_id)
        start_time_list.append(start)
        end_time_list.append(end)
    df['item_id'] = item_id_list
    df['start_time'] = start_time_list
    df['end_time'] = end_time_list
    df['diff'] = df['end_time'] - df['start_time']
    df['start_time'] = df['start_time'].apply(lambda x: time.strftime('%Y%m%d%H', time.localtime(x)))
    df['end_time'] = df['end_time'].apply(lambda x: time.strftime('%Y%m%d%H', time.localtime(x)))

    df_dummy = pd.get_dummies(all_news_info['cate_id'], prefix='cate')
    df_dummy = pd.concat([all_news_info[['item_id', 'timestamp']], df_dummy], axis=1, join='inner')

    df = pd.merge(df_dummy, df, on='item_id')
    df['timestamp'] = df['timestamp'].apply(lambda x: time.strftime('%Y%m%d%H', time.localtime(x)))

    return df


def show_user_cate(train):
    cate_list = ['1_1',
     '1_10',
     '1_11',
     '1_12',
     '1_13',
     '1_14',
     '1_15',
     '1_16',
     '1_17',
     '1_18',
     '1_19',
     '1_2',
     '1_23',
     '1_27',
     '1_28',
     '1_3',
     '1_4',
     '1_5',
     '1_6',
     '1_7',
     '1_8',
     '1_9',
     '3_1',
     '3_13',
     '3_15',
     '3_19',
     '3_2',
     '3_20',
     '3_23',
     '3_3',
     '3_6',
     '3_7',
     '3_8',
     '3_9',
     '4_1',
     '4_5',
     '6_1']
    train = train[['user_id', 'cate_id', 'action_type']].groupby(['user_id', 'cate_id']).count().unstack().fillna(0)
    train.columns = ["cate_" + cate_id for cate_id in cate_list]
    train = train.reset_index()
    return train

# def dayTop5( train,days=[16,17,18] ):
#     train['day'] = train['action_time'].apply( lambda x:time.strftime('%d'),time.localtime(x)  )
#     top_list = []
#     d_list = []
#     for d in days:
#         d_top5 = train[ train['day']==d ].groupby( ['item_id',],as_index=False ).count()[['item_id','day']].sort_values(
#             ['day'],ascending=False).head(5)['item_id'].values
#         d_list = d_list.extend( [d]*5 )
#         top_list = top_list.extend( list( d_top5 ) )
#     pd.DataFrame()
#     train['item_id'].apply(  )


def make_train_set():
    # test = pd.read_csv('../data/test.csv')
    train = pd.read_csv('../data/train.csv')
    item = pd.read_csv('../data/news_info.csv')
    label = randomSelectTrain(train)

    user_feat = show_user_cate(train)
    df = pd.merge(label, user_feat, on='user_id')

    item_index = item.reset_index()[['item_id','index']]
    df = pd.merge(df, item_index, on='item_id')
    index = df[['user_id','item_id']]
    label = df[['label']]
    data = df.drop(['user_id','item_id','label'],axis=1)
    return index,label,data

def xgb_train( ):
    i = 0
    print('=========================training==============================')
    index,label,data = make_train_set()
    print label.head()
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1990)
    # see how many neg/pos sample
    label = label.values
    print( "neg:{0},pos:{1}".format(len(label[label == 0]), len(label[label == 1])) )
    # scale_pos_weight = (len(label[label == 0])) / float(len(label[label == 1]))

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 3,
             'min_child_weight': 2, 'gamma': 0, 'subsample': 0.6, 'colsample_bytree': 0.8,
              'eta': 0.1, 'lambda': 5,  # L2惩罚系数 'scale_pos_weight': scale_pos_weight,
             'objective': 'binary:logistic', 'eval_metric': 'auc',
             'early_stopping_rounds': 100,  # eval 得分没有继续优化 就停止了
             'seed': 1990, 'nthread': 4, 'silent': 1
             }
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_boost_round=50, evals=evallist)

    bst.save_model(os.path.join('../model', ('xgb%02d.model' % i)))
    print('save feature score and feature information')
    feature_score = bst.get_fscore()
    for key in feature_score:
        feature_score[key] = [feature_score[key]]
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))
    if not os.path.exists('../data/features'):
        os.mkdir('../data/features')

    fpath = os.path.join('../data/features', 'feature_score%02d.csv' % i)
    with open(fpath, 'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)

xgb_train()