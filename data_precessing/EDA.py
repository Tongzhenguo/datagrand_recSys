# coding=utf-8
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
#候选咨询
import time
import time
import numpy as np

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def time_stats(  ):
    tr = pd.read_csv('data/train.csv')
    tr['date'] = tr['action_time'].apply(lambda x: time.strftime('%Y%m%d%H', time.localtime(x)))
    tr[['date', 'user_id']].drop_duplicates().groupby(['date'], as_index=False).count().sort_values(['date']).to_csv(
        'data/date_uv.csv', index=False)

    user_last_date = tr[['action_time', 'user_id']].groupby(['user_id'], as_index=False).max()
    user_early_date = tr[['action_time', 'user_id']].groupby(['user_id'], as_index=False).min()
    user_dur = pd.merge(user_early_date, user_last_date, on='user_id')
    user_dur['dur'] = user_dur['action_time_y'] - user_dur['action_time_x']
    user_dur.sort_values(['dur'], ascending=False)[['user_id', 'dur']].to_csv('data/user_dur.csv', index=False)

    tr['pop'] = tr['action_time'].apply(lambda t: 1 / (1.0 + 0.1 * (1487433599 - t)))
    item_pop = tr[['item_id', 'pop']].groupby(['item_id'], as_index=False).sum()
    item_pop.to_csv('data/item_pop.csv', index=False)

    item_pop['pop_lev'] = item_pop['pop'].apply( lambda x:int( 20*( x - 0.000039 ) / ( 4.962493 - 0.000039 ) ) )
    item_pop[['item_id','pop_lev']].to_csv('data/item_pop_lev.csv', index=False)

    item_last_date = tr[['action_time', 'item_id']].groupby(['item_id'], as_index=False).max()
    item_early_date = tr[['action_time', 'item_id']].groupby(['item_id'], as_index=False).min()
    item_dur = pd.merge(item_early_date, item_last_date, on='item_id')
    item_dur['dur'] = item_dur['action_time_y'] - item_dur['action_time_x']

    pop_lev_dur = pd.merge( item_pop[['item_id','pop_lev']],item_dur[['item_id','dur']],on='item_id' ).groupby( ['pop_lev'],as_index=False
            ).mean()[['pop_lev','dur']]
    pop_lev_dur.to_csv( 'data/pop_lev_dur.csv',index=False )

def do_test():
    test_csv =  pd.DataFrame()
    user_list = []
    item_list = []
    test = pd.read_csv('../data/test.txt')
    for u,group in test.groupby(['user_id'],as_index=False):
        iid_list = group['item_id'].values[0].split(" ")
        user_list.extend( [u]*len(iid_list) )
        item_list.extend(iid_list)
    test_csv['user_id'] = user_list
    test_csv['item_id'] = item_list
    test_csv.to_csv('../data/test.csv',index=False)

def desc():
    news = pd.read_csv('../data/news_info.csv')
    print( len(news) )#41252

    #资讯类别
    print( len( news['cate_id'].unique() ) ) #37


    train = pd.read_csv('../data/train.csv')
    print( len( train['item_id'].unique() ) )#65214
    hot_cate = train[['cate_id','user_id']].groupby(['cate_id'],as_index=False).count().sort_values(['user_id'],ascending=False).head()
    print( list(hot_cate['cate_id'] ) ) #['1_1', '1_6', '1_3', '1_11', '1_2']
    hot_item = train[['user_id','item_id']].groupby(['item_id'],as_index=False).count().sort_values(['user_id'],ascending=False).head()
    print( list(hot_item['item_id'] ) )#[543083, 549608, 542524, 540198, 550192]


    #行为类型
    print( ( train['action_type'].unique() ) )#['view' 'deep_view' 'share' 'collect' 'comment']
    print( float(len( train[ train['action_type']=='view' ]))  / len(train[ train['action_type']=='deep_view' ] ) ) #1.26020722627
    print(len(train[train['action_type'] == 'view']) / len(train[train['action_type'] == 'share'])) #3955
    print(len(train[train['action_type'] == 'view']) / len(train[train['action_type'] == 'collect'])) #166
    print(len(train[train['action_type'] == 'view']) / len(train[train['action_type'] == 'comment'])) #420



    #最新资讯
    all_items = set( news['item_id'].unique() )
    train_items = set( train['item_id'].unique() )
    new_items = all_items - train_items
    new_items = list(new_items)
    print( len(new_items) ) #9999
    # print( new_items[:5]

    #待推荐用户
    users = pd.read_csv('../data/candidate.txt')
    print( len( users ) )#28501

    #用户的冷启动问题
    print( ('约' + str(len(users) - len( pd.merge(train.drop_duplicates('user_id') ,users,on='user_id' ) )) + '用户存在冷启动问题')  )# 0

def show_item_display_time( ):
    train = pd.read_csv('../data/train.csv')
    item_id_list = []
    start_time_list = []
    end_time_list = []
    df = pd.DataFrame()
    for  item_id,group in  train.groupby(['item_id'],as_index=False):
        start = group['action_time'].min()
        end = group['action_time'].max()
        # pv = group['view'].count()
        item_id_list.append( item_id )
        start_time_list.append( start  )
        end_time_list.append( end )
    df['item_id'] = item_id_list
    df['start_time'] = start_time_list
    df['end_time'] = end_time_list
    df['diff'] = df['end_time'] - df['start_time']
    df['start_time'] = df['start_time'].apply( lambda x:time.strftime( '%Y%m%d %H:%M:%S',time.localtime(x) ) )
    df['end_time'] = df['end_time'].apply(lambda x: time.strftime('%Y%m%d %H:%M:%S',time.localtime(x)))

    news_info = pd.read_csv('../data/news_info.csv')
    df = pd.merge( news_info,df,on='item_id' )
    df['timestamp'] = df['timestamp'].apply( lambda x:time.strftime( '%Y%m%d %H:%M:%S',time.localtime(x) ))
    df.to_csv( '../data/item_display.csv',index=False )
    # plt.show()

def show_ds_pv():
    train = pd.read_csv('../data/train.csv')
    start = train['action_time'].min()
    train['ds'] = train['action_time'].apply( lambda x: start+( x-start )/3600*3600 )
    df = train[['user_id','item_id','ds']].groupby( ['item_id','ds'],as_index=False ).count()
    df.columns = ['item_id','ds','pv']
    df = df[['item_id','ds','pv']].sort_values( ['item_id','ds','pv'] )
    df['ds'] = df['ds'].apply( lambda x: time.strftime('%Y%m%d %H:%M:%S',time.localtime(x) ) )
    df.to_csv( '../data/item_ds_pv.csv',index=False )


def show_cate_diff():
    item = pd.read_csv('../data/news_info.csv')
    item_display = pd.read_csv('../data/item_display.csv')
    item_ds_pv = pd.read_csv('../data/item_ds_pv.csv')

    df_mean = pd.merge(item, item_display, on='item_id')[['cate_id', 'diff']].groupby(
        ['cate_id'],as_index=False).mean()
    df_min = pd.merge(item, item_display, on='item_id')[['cate_id', 'diff']].groupby(
        ['cate_id'],as_index=False).min()
    df_max = pd.merge(item, item_display, on='item_id')[['cate_id', 'diff']].groupby(
        ['cate_id'],as_index=False).max()
    df_end_time = pd.merge(item, item_display, on='item_id')[['cate_id', 'end_time']].groupby(
        ['cate_id'],as_index=False).max()
    df = pd.merge( df_mean,df_min,on='cate_id' )
    df = pd.merge( df,df_max,on='cate_id' )
    df = pd.merge(df, df_end_time, on='cate_id')
    df.columns = ['cate_id','df_mean','df_min','df_max','df_end_time']
    df['df_mean'] = df['df_mean'].apply(int)
    df['df_min'] = df['df_min'].apply(int)
    df['df_max'] = df['df_max'].apply(int)
    cate_pv = pd.merge(item, item_ds_pv,on='item_id')[['cate_id', 'pv']].groupby(['cate_id'],as_index=False).sum()
    df = pd.merge( df,cate_pv,on='cate_id' )
    df.sort_values(['pv','df_mean','df_max','df_end_time'],ascending=False).to_csv('../data/cate_desc.csv',index=False)

def show_hh_pv():
    pv = pd.read_csv('../data/item_ds_pv.csv')
    pv['hh'] = pd.to_datetime(pv['ds'], format='%Y%m%d %H:%M:%S')
    pv['hh'] = pv['hh'].apply( lambda x:x.hour )
    hh_pv = pv.groupby(['hh'], as_index=False)[[ 'pv']].sum().sort_values(['hh'])
    hh_pv.to_csv('../data/hh_pv.csv', index=False)
    plt.bar(range(0,24),list(hh_pv['pv'].values),fc='g')
    plt.show() #高峰期：6-9，19-22

def show_user_cate( train ):
    train = train[['user_id','cate_id','action_type']].groupby(['user_id','cate_id']).count().instack().fillna(0)
    train.columns = [ "cate_"+cate_id for cate_id in list(train.columns) ]
    train = train.reset_index()
    return train

def show_item_cate_dummy( all_news_info ):
    df_dummy =pd.get_dummies( all_news_info['cate_id'],prefix='cate' )
    df = pd.concat( [ all_news_info[['item_id','timestamp']],df_dummy ],axis=1,join='inner' )
    return df

# def show_item_pv_hh( train,hour_list=[1,2,3,4,5,6,12,24,72] ):
#     for i in hour_list:
#         time.mktime( time.strftime('') )
#         train[ train['action_time']> ]

if __name__ == "__main__":
    # desc()
    # show_ds_pv()
    # show_item_display_time()
    # show_cate_diff()
    # show_hh_pv()
    do_test()