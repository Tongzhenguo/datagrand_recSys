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
    正负类选取：用户看过的是正类，负类是从所有物品集随机抽取的，物品集没有对物品去重，故被关注多的商品被选中的概率大
    特征工程：
        用户端有对各品类的交互次数，
        物品端有
           发布时间，
           index,
           阅读次数，
           3种流行度算法的得分，
           隐性评分的avg,max,min,std,中位数

"""

def randomSelectTrain( train,flag ):
    path = '../cache/train_{flag}.pkl'.format( flag=flag )
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
        for user, group in train.groupby(['user_id']):
            print( len(user_list) )
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
    cate_list = list( train['cate_id'].unique() )
    cate_list.sort()
    train = train[['user_id', 'cate_id', 'action_type']].groupby(['user_id', 'cate_id']).count().unstack().fillna(0)
    train.columns = ["cate_" + cate_id for cate_id in cate_list]
    train = train.reset_index()
    return train

def get_item_action_count( train ):
    train = train[[ 'item_id','user_id' ]].groupby( ['item_id'],as_index=False ).count()
    train.columns = ['item_id','action_count']
    return train

def get_latent_factor_product( df ):
    print(' 读取用户和物品矩阵 ')
    user_mat = pd.read_csv('../data/user_mat.csv')
    item_mat = pd.read_csv('../data/item_mat.csv')
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
    df = pd.merge(df, user_mat, on='user_id')
    df = pd.merge( df,item_mat,on='item_id' )
    feat = ['user_id','item_id']+[ "factor_product_"+str(i) for i in range(35) ]
    for i in range(35):
        df[ "factor_product_"+str(i) ] = df[ 'factor_'+str(i)+'_x' ] * df[ 'factor_'+str(i)+'_y' ]
    return df[ feat ]

def get_pop( tr ):
    tr['pop'] = tr['action_time'].apply(lambda t: 1 / (1.0 + 0.1 * (1487433599 - t)))
    item_pop = tr[['item_id', 'pop']].groupby(['item_id'], as_index=False).sum()
    return item_pop

def get_hacker_news( tr,item ):
    item_action_cnt = \
    tr[['user_id', 'item_id', 'action_type']].drop_duplicates().groupby(['item_id'], as_index=False).count()[
        ['item_id', 'action_type']]
    item_action_cnt.columns = ['item_id', 'action_cnt']
    item_pop = pd.merge(item[['item_id', 'timestamp']], tr, on='item_id')
    item_pop = pd.merge(item_action_cnt, item_pop, on='item_id')
    item_pop['pop'] = item_pop['action_cnt'] / pow((item_pop['action_time'] - item_pop['timestamp']) / 3600,
                                                   5.8)  # 5.8等于10.8，优于1.8,2.8
    item_pop = item_pop[['item_id', 'pop']].groupby(['item_id'], as_index=False).sum()
    return item_pop

def get_action_weight( x):
    if x == 'view': return 1
    if x == 'deep_view': return 2
    if x == 'share':return 8
    if x == 'comment': return 6
    if x == 'collect':return 5
    else:return 1

def get_item_rat_stats(tr):
    tr['rat'] = tr['action_type'].apply( get_action_weight )
    tmp = tr[['user_id', 'item_id', 'rat']].drop_duplicates().groupby(['item_id'], as_index=False)
    rat_avg = tmp.mean()[['item_id', 'rat']]
    rat_max = tmp.max()[['item_id', 'rat']]
    rat_min = tmp.min()[['item_id', 'rat']]
    rat_sum = tmp.sum()[['item_id', 'rat']]
    df = pd.merge( rat_max,rat_min,on='item_id')
    df.columns = ['item_id','rat_max','rat_min']
    df['rat_mid'] = df['rat_max'] + df['rat_min']
    df['rat_mid'] /= 2
    df = pd.merge(df, rat_avg, on='item_id')
    df = pd.merge(df, rat_sum, on='item_id')
    df.columns = [ 'rat_max','rat_min','rat_mid','rat_avg','item_id','rat_sum']
    return df


def make_train_set():
    train = pd.read_csv('../data/train.csv')
    start_date = time.mktime(time.strptime('2017-2-18 00:00:00', '%Y-%m-%d %H:%M:%S'))
    end_date = time.mktime(time.strptime('2017-2-18 12:00:00', '%Y-%m-%d %H:%M:%S'))
    train = train[(train.action_time <= end_date) & (train.action_time>= start_date)]
    item = pd.read_csv('../data/news_info.csv')
    user = pd.read_csv('../data/candidate.txt')
    train = pd.merge( user,train,on='user_id' )
    label = randomSelectTrain(train,"train_all_item")

    user_feat = show_user_cate(train)
    df = pd.merge(label, user_feat, on='user_id')
    item_action_count = get_item_action_count(train)
    df = pd.merge( df,item_action_count,on='item_id' )
    latent_factor_product = get_latent_factor_product(df)
    df = pd.merge(df, latent_factor_product, on=( 'user_id','item_id' ))
    item_pop = get_pop( train )
    df = pd.merge( df,item_pop,on='item_id' )
    item_pop_news = get_hacker_news(train, item)
    df = pd.merge( df,item_pop_news,on='item_id' )
    rat_stats = get_item_rat_stats(train)
    df = pd.merge(df, rat_stats)

    item_index = item.reset_index()[['item_id','index','timestamp']]
    df = pd.merge(df, item_index, on='item_id')
    index = df[['user_id','item_id']]
    label = df[['label']]
    data = df.drop(['user_id','item_id','label'],axis=1)
    return index,label,data

def make_test_data():
    train = pd.read_csv('../data/train.csv')
    train = train[train.action_time >= time.mktime(time.strptime('2017-2-18 12:00:00', '%Y-%m-%d %H:%M:%S'))]
    item = pd.read_csv('../data/news_info.csv')
    user = pd.read_csv('../data/candidate.txt')
    train = pd.merge(user, train, on='user_id')
    train = pd.merge( item[['item_id']],train,on='item_id' )
    label = randomSelectTrain(train,"test_item")

    user_feat = show_user_cate(train)
    df = pd.merge(label[ label['label']==0 ], user_feat, on='user_id')
    item_action_count = get_item_action_count(train)
    df = pd.merge(df, item_action_count, on='item_id')
    latent_factor_product = get_latent_factor_product(df)
    df = pd.merge(df, latent_factor_product, on=('user_id', 'item_id'))
    item_pop = get_pop(train)
    df = pd.merge(df, item_pop, on='item_id')
    item_pop_news = get_hacker_news(train, item)
    df = pd.merge( df,item_pop_news,on='item_id' )
    rat_stats = get_item_rat_stats(train)
    df = pd.merge(df, rat_stats)

    item_index = item.reset_index()[['item_id', 'index','timestamp']]
    df = pd.merge(df, item_index, on='item_id')
    index = df[['user_id', 'item_id']]
    data = df.drop(['user_id', 'item_id', 'label'], axis=1)
    return index, data

def xgb_train( ):
    i = 0
    print('=========================training==============================')
    index,label,data = make_train_set()
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