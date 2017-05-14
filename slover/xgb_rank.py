# coding=utf-8
import pandas as pd
import time


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
    train = train[['user_id', 'cate_id', 'action_type']].groupby(['user_id', 'cate_id']).count().instack().fillna(0)
    train.columns = ["cate_" + cate_id for cate_id in list(train.columns)]
    train = train.reset_index()
    return train


def get_label(train):
    train = train[['user_id', 'item_id']]
    # 默认每个用户推最火的20个资讯
    news = train[['item_id']].drop_duplicates()
    train['label'] = 1



def make_train_set():
    train = pd.read_csv('../data/train.csv')
    news_info = pd.read_csv('../data/all_news_info.csv')
    end = time.mktime(time.strptime('2017-2-17 00:00:00', '%Y-%m-%d %H:%M:%S'))
    train = train[train['action_time'] < end]
    item_feat = show_item_display_time(train, news_info)
    user_feat = show_user_cate(train)
