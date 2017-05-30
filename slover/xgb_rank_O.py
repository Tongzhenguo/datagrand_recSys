# coding=utf-8
import os
import random
import gc
import pandas as pd
import time
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb

"""
    正负类选取：
        用户看过的是正类，负类是从所有物品集随机抽取的，物品集没有对物品去重，故被关注多的商品被选中的概率大
        recent k items
    特征工程：
        用户端有
            对各品类的交互次数，
            用户各行为统计
            用户view占总用户view的比例
        物品端有
           发布时间，
           阅读次数:分时段

"""

def randomSelectTrain( train,start,end ):
    path = '../cache/train_{st}_{ed}.pkl'.format( st=start,ed=end )
    if os.path.exists( path ):
        data = pickle.load(open(path, "rb"))
    else:
        print('抽样选择与用户阅读资讯相同数量的热门资讯')
        data = pd.DataFrame()
        user_list = []
        item_list = []
        label_list = []
        items_pool = list(train['item_id'].values)
        rec = dict()
        print 'recent k items'
        train = train.sort_values(['action_time'],ascending=False)[['user_id','item_id']].drop_duplicates()
        for user, group in train.groupby(['user_id'],sort=False):
            viewed_item = list( group['item_id'].head(5).values )
            for item in viewed_item:
                rec[item] = 1
                user_list.append(user)
                item_list.append(item)
                label_list.append(1)
            n = 0
            for i in range( 0, len(items_pool) ):
                item = items_pool[random.randint(0, len(items_pool) - 1)]
                if item in list(group['item_id'].values): #fix
                    continue
                user_list.append(user)
                item_list.append(item)
                label_list.append(0)
                n += 1
                rec[item] = 0
                if n > len(viewed_item):
                    break
            rec.clear()
        data['user_id'] = user_list
        data['item_id'] = item_list
        data['label'] = label_list
        del user_list
        del item_list
        del label_list
        gc.collect()
        pickle.dump( data,open(path,'wb'),True )
    return data


def show_user_cate(train):
    cate_list = list( train['cate_id'].unique() )
    cate_list.sort()
    train = train[['user_id', 'cate_id', 'action_type']].groupby(['user_id', 'cate_id']).count().unstack().fillna(0)
    train.columns = ["cate_" + cate_id for cate_id in cate_list]
    train = train.reset_index()
    return train

def get_user_action_count( train ):
    action_list = list( train['action_type'].unique() )
    action_list.sort()
    train = train[['user_id','action_type','action_time']].groupby(['user_id','action_type']).count().unstack().fillna(0)
    train.columns = [str(t)+'_count' for t in action_list ]
    train = train.reset_index()
    return train

def get_user_view_ratio( tr ):
    tr = tr[ tr.action_type=='view' ]
    tr = tr[['user_id','action_type']].groupby( ['user_id'],as_index=False ).count()
    le = len( tr )
    tr['view_ratio'] = tr['action_type'].apply( lambda x:x/float(le) )
    return tr[['user_id','view_ratio']]

def get_item_action_count(  ):
    train = pd.read_csv('../data/train.csv')
    item = pd.read_csv('../data/news_info.csv')
    res = pd.DataFrame()
    res['item_id'] = item['item_id']
    for w in [ ['2017-2-16 06:00:00','2017-2-16 09:00:00'],
               ['2017-2-16 09:00:00', '2017-2-16 19:00:00'],
               ['2017-2-16 19:00:00', '2017-2-16 22:00:00'],
               ['2017-2-17 06:00:00', '2017-2-17 09:00:00'],
               ['2017-2-17 09:00:00', '2017-2-17 19:00:00'],
               ['2017-2-17 19:00:00', '2017-2-17 22:00:00'],
               ['2017-2-18 06:00:00', '2017-2-18 09:00:00'],
               ['2017-2-18 09:00:00', '2017-2-18 19:00:00'],
               ['2017-2-18 19:00:00', '2017-2-18 22:00:00'],
               ]:

        start_date = time.mktime(time.strptime(w[0], '%Y-%m-%d %H:%M:%S'))
        end_date = time.mktime(time.strptime(w[1], '%Y-%m-%d %H:%M:%S'))
        tmp = train[(train.action_time < end_date) & (train.action_time>= start_date)]
        tmp = tmp[[ 'item_id','user_id' ]].groupby( ['item_id'],as_index=False ).count()
        tmp.columns = ['item_id','action_count_{st}_{ed}'.format(st=start_date,ed=end_date)]
        res = pd.merge( res,tmp,how='left',on='item_id' ).fillna(0)
    return res


def get_action_weight( x):
    if x == 'view': return 1
    if x == 'deep_view': return 2
    if x == 'share':return 8
    if x == 'comment': return 6
    if x == 'collect':return 5
    else:return 1


def make_train_set():
    train = pd.read_csv('../data/train.csv')
    start_date = time.mktime(time.strptime('2017-2-16 06:00:00', '%Y-%m-%d %H:%M:%S'))
    end_date = time.mktime(time.strptime('2017-2-16 09:00:00', '%Y-%m-%d %H:%M:%S'))
    train = train[(train.action_time <= end_date) & (train.action_time>= start_date)]
    # item = pd.read_csv('../data/news_info.csv')
    user = pd.read_csv('../data/candidate.txt')
    train = pd.merge( user,train,on='user_id' )
    label = randomSelectTrain(train,start_date,end_date)

    user_feat = show_user_cate(train)
    df = pd.merge(label, user_feat, on='user_id')
    uac = get_user_action_count( train)
    df = pd.merge(df, uac, on='user_id')
    uwr = get_user_view_ratio( train)
    df = pd.merge(df, uwr, on='user_id')
    item_action_count = get_item_action_count()
    df = pd.merge( df,item_action_count,on='item_id' )

    index = df[['user_id','item_id']]
    label = df[['label']]
    data = df.drop(['user_id','item_id','label'],axis=1)
    return index,label,data

def make_test_data():
    train = pd.read_csv('../data/train.csv')
    start_date = time.mktime(time.strptime('2017-2-18 19:00:00', '%Y-%m-%d %H:%M:%S'))
    end_date = time.mktime(time.strptime('2017-2-18 22:00:00', '%Y-%m-%d %H:%M:%S'))
    train = train[(train.action_time >= start_date) & (train.action_time <= end_date)]
    item = pd.read_csv('../data/news_info.csv')
    user = pd.read_csv('../data/candidate.txt')
    train = pd.merge(user, train, on='user_id')
    train = pd.merge( item[['item_id']],train,on='item_id' )
    label = randomSelectTrain(train,start_date,end_date)

    user_feat = show_user_cate(train)
    df = pd.merge(label[ label['label']==0 ], user_feat, on='user_id')
    uac = get_user_action_count( train )
    df = pd.merge(df, uac, on='user_id')
    uwr = get_user_view_ratio( train)
    df = pd.merge(df, uwr, on='user_id')
    item_action_count = get_item_action_count()
    df = pd.merge(df, item_action_count, on='item_id')

    index = df[['user_id', 'item_id']]
    data = df.drop(['user_id', 'item_id', 'label'], axis=1)
    return index, data

def xgb_train( ):
    i = 0
    print('=========================training==============================')
    index,label,data = make_train_set(  )
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
    bst = xgb.train(param, dtrain, num_boost_round=500, evals=evallist)

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


def xgb_sub(  ):
    i = 0
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model( '../model/xgb%02d.model' % i )  # load data

    print('开始构造测试集---------------------------------------------------')
    sub_index, sub_trainning_data = make_test_data()
    test = xgb.DMatrix(sub_trainning_data)
    sub_index['score'] = bst.predict(test)
    sub_index = sub_index.sort_values( ['user_id','score'],ascending=False )
    rec = pd.DataFrame()
    user_list = []
    rec_items_list = []
    for user,group in sub_index.groupby( ['user_id'],as_index=False,sort=False ):
        rec_items = " ".join(map(str, list(group['item_id'].head(5).values)))
        user_list.append(user)
        rec_items_list.append(rec_items)
    rec['user_id'] = user_list
    rec['item_id'] = rec_items_list

    print('还有部分的冷启动用户,推荐test中的topHot5')  # 这里也可改进
    users = pd.read_csv('../data/candidate.txt')
    test = pd.read_csv('../data/test.csv')
    topHot = \
        test.groupby(['item_id'], as_index=False).count().sort_values(['user_id'], ascending=False).head(5)[
        'item_id'].values
    oldrec = users
    oldrec['oldrec_item'] = [" ".join( map( str,list(topHot) ) )] * len(oldrec)
    oldrec = pd.merge(oldrec, rec, how='left', on='user_id', ).fillna(0)
    oldrec = oldrec[oldrec.item_id == 0][['user_id', 'oldrec_item']]
    oldrec.columns = ['user_id', 'item_id']
    rec = rec.append(oldrec)

    rec = rec.drop_duplicates('user_id')
    rec.to_csv('../result/result20170505.csv', index=None,header=None)

if __name__ == '__main__':
    xgb_train()
    xgb_sub() #加了pop和隐因子反而下降了