# coding=utf-8
import pandas as pd

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
print new_items[:5]

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


if __name__ == "__main__":
    show_new_user_item_count()