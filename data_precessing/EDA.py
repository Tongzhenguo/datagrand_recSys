# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
#候选咨询
news = pd.read_csv('../data/news_info.csv')
# print len(news) #56342

#资讯类别
# print len( news['cate_id'].unique() ) #37


train = pd.read_csv('../data/train.csv')
# print len( train['item_id'].unique() ) #41249
hot_cate = train[['cate_id','user_id']].groupby(['cate_id'],as_index=False).count().sort_values(['user_id'],ascending=False).head()
# print list(hot_cate['cate_id'] ) #['1_1', '1_6', '1_3', '1_11', '1_2']
hot_item = train[['user_id','item_id']].groupby(['item_id'],as_index=False).count().sort_values(['user_id'],ascending=False).head()
# print list(hot_item['item_id'] ) #[549608, 550192, 550039, 542524, 542113]


#行为类型
# print ( train['action_type'].unique() ) #['view' 'deep_view' 'share' 'collect' 'comment']


#最新资讯
all_items = set( news['item_id'].unique() )
train_items = set( train['item_id'].unique() )
new_items = all_items - train_items
new_items = list(new_items)
# print len(new_items) #15093
# print new_items[:5]

#待推荐用户
users = pd.read_csv('../data/candidate.txt')
# print len( users ) #29999

def show_new_user_item_count():
    #用户的冷启动问题比物品的冷启动问题严重
    print len(news) #56342
    print len(pd.merge(train.drop_duplicates('item_id'), news, on='item_id')) #41249
    print ( '约'+str( len(news)-len(pd.merge(train.drop_duplicates('item_id'), news, on='item_id')) )+'资讯存在冷启动问题' ) #15093

    print len(users)  # 30000
    print len( pd.merge(train.drop_duplicates('user_id') ,users,on='user_id' ) )  # 17507
    print (
    '约' + str(len(users) - len( pd.merge(train.drop_duplicates('user_id') ,users,on='user_id' ) )) + '用户存在冷启动问题')  # 12493

def show_item_display_time( ):
    train = pd.read_csv('../data/train.csv')
    item_id_list = []
    start_time_list = []
    end_time_list = []
    # pv_list = []
    df = pd.DataFrame()
    for  item_id,group in  train.groupby(['item_id'],as_index=False):
        start = group['action_time'].min()
        end = group['action_time'].max()
        # pv = group['view'].count()
        item_id_list.append( item_id )
        start_time_list.append( start )
        end_time_list.append( end )
    df['item_id'] = item_id_list
    df['start_time'] = start_time_list
    df['end_time'] = end_time_list
    df['diff'] = df['end_time'] - df['start_time']
    df.to_csv( '../data/item_display.csv',index=False )
    df['diff'].plot()
    plt.show()

def show_item_display_time():
    train = pd.read_csv('../data/train.csv')
    start = train['action_time'].min()
    end = train['action_time'].max()
    train['ds'] = train['action_time'].apply( lambda x: start+( x-start )/3600*3600 )
    df = train[['user_id','item_id','ds']].groupby( ['item_id','ds'],as_index=False ).count()
    df.columns = ['item_id','ds','pv']
    df[['item_id','ds','pv']].sort_values( ['item_id','ds','pv'] ).to_csv( '../data/item_ds_pv.csv',index=False )



if __name__ == "__main__":
    # show_new_user_item_count()
    show_item_display_time()