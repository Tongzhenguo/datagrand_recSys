# coding=utf-8
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
#候选咨询
import time

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def desc():
    news = pd.read_csv('../data/news_info.csv')
    # print( len(news) )#41252

    #资讯类别
    # print( len( news['cate_id'].unique() ) ) #37


    train = pd.read_csv('../data/train.csv')
    print( len( train['item_id'].unique() ) )#65214
    hot_cate = train[['cate_id','user_id']].groupby(['cate_id'],as_index=False).count().sort_values(['user_id'],ascending=False).head()
    print( list(hot_cate['cate_id'] ) ) #['1_1', '1_6', '1_3', '1_11', '1_2']
    hot_item = train[['user_id','item_id']].groupby(['item_id'],as_index=False).count().sort_values(['user_id'],ascending=False).head()
    print( list(hot_item['item_id'] ) )#[543083, 549608, 542524, 540198, 550192]


    #行为类型
    print( ( train['action_type'].unique() ) )#['view' 'deep_view' 'share' 'collect' 'comment']


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

    #用户的冷启动问题比物品的冷启动问题严重
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
    end = train['action_time'].max()
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



if __name__ == "__main__":
    # show_ds_pv()
    show_item_display_time()
    # show_cate_diff()
    # show_hh_pv()