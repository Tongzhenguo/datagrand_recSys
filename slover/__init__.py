# coding=utf-8
import pandas as pd
import time

# 2种流行度排名，
#            隐性评分的avg,max,min,std,中位数
#            滑动窗口，recent k items
#         35个隐变量的乘积


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
    item_pop['time_rank'] = item_pop['pop'].rank()
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
    item_pop['pop_rank'] = item_pop['pop'].rank()
    return item_pop

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

# latent_factor_product = get_latent_factor_product(df)
# df = pd.merge(df, latent_factor_product, on=( 'user_id','item_id' ))
# item_pop = get_pop( train )
# df = pd.merge( df,item_pop,on='item_id' )
# item_pop_news = get_hacker_news(train, item)
# df = pd.merge( df,item_pop_news,on='item_id' )
# rat_stats = get_item_rat_stats(train)
# df = pd.merge(df, rat_stats)



