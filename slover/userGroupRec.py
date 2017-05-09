# coding=utf-8
"""
    先对用户分组，然后对每一个分组推荐top5
    第一个策略：将关注同一cate最多的做成一组，为这组推荐cate下最火的5个item :0.007774
"""
import pandas as pd

users = pd.read_csv('../data/candidate.txt')
train = pd.read_csv('../data/train.csv')
item = pd.read_csv('../data/news_info.csv')

user_group = pd.merge( users,train,on='user_id')[['user_id','cate_id','item_id']]\
    .groupby(['user_id','cate_id'],as_index=False).count().sort_values(['item_id'],ascending=False).drop_duplicates('user_id')[['user_id','cate_id']]

cate_item_count = train[['user_id','cate_id','item_id']]\
    .groupby(['cate_id','item_id'],as_index=False).count().sort_values(['user_id'],ascending=False)
cate_list = []
cate_rec_list = []
cate_rec = pd.DataFrame(  )
for c in list(train['cate_id'].unique() ):
    cate_item_count = pd.merge( cate_item_count,item[['item_id']],on='item_id' )
    cate_top5 = cate_item_count[ cate_item_count.cate_id == c ]['item_id'].head(5).values
    cate_list.append( c )
    cate_rec_list.append(  " ".join( map( str, list( cate_top5 )) ) )
cate_rec['cate_id'] = cate_list
cate_rec['rec_item'] = cate_rec_list

rec = pd.merge( user_group,cate_rec,on='cate_id')[['user_id','rec_item']]
# print len( rec )
#还有1w的冷启动用户,直接推荐全局最热的
oldrec = pd.read_csv('../result/result_0.007403.csv')
oldrec = pd.merge(  oldrec,rec,how='left',on='user_id', ).fillna(0)
oldrec = oldrec[ oldrec.rec_item==0 ][['user_id','oldrec_item']]
oldrec.columns = [ 'user_id', 'rec_item']
rec = rec.append( oldrec )
rec.to_csv('../result/result.csv',index=None,header=None)
