# coding=utf-8
"""
    为用户推荐最热的咨询，如果用户以前看过不会再推荐
    根据测试集中的行为数据去搞
"""
import pandas as pd

def r1():
    train = pd.read_csv('../data/train.csv')
    hot_item = train[['user_id','item_id']].groupby(['item_id'],as_index=False).count().sort_values(['user_id'],ascending=False)
    hot_item.columns = ['item_id','count']
    hot_item = list( hot_item['item_id'].values )[:5]

    users = pd.read_csv('../data/candidate.txt')
    recItems = " ".join( map( str, hot_item) )
    # result = open('../result/result.csv','wb')
    users['item_id'] = recItems
    users.to_csv('../result/result.csv',index=False)

def r2():
    train = pd.read_csv('../data/train.csv')
    hot_item = train[['user_id', 'item_id']].groupby(['item_id'], as_index=False).count().sort_values(['user_id'],
                                                                                                      ascending=False)
    hot_item.columns = ['item_id', 'count']
    hot_item = list(hot_item['item_id'].values)[:100]
    recItems = " ".join(map(str, hot_item))
    users = pd.read_csv('../data/candidate.txt')
    users['item_id'] = recItems

    test = pd.read_csv('../data/test.txt')
    rec = pd.merge( users,test,how='left',on='user_id' ).fillna("")
    rec['item_id'] = rec['item_id_x'] + "," + rec['item_id_y']
    rec['item_id'] = rec['item_id'].apply(help)
    rec[['user_id','item_id']].drop_duplicates("user_id").to_csv('../result/result.csv', index=False,header=False)

def r3():
    filterItems = pd.read_csv('../data/filterItems.csv')
    hot_item = filterItems[['user_id', 'item_id']].groupby(['item_id'], as_index=False).count().sort_values(['user_id'],
                                                                                                ascending=False)
    hot_item = list(hot_item['item_id'].values)[:10]
    recItems = " ".join(map(str, hot_item))
    users = pd.read_csv('../data/candidate.txt')
    users['item_id'] = recItems

    test = pd.read_csv('../data/test.txt')
    rec = pd.merge(users, test, how='left', on='user_id').fillna("")
    rec['item_id'] = rec['item_id_x'] + "," + rec['item_id_y']
    rec['item_id'] = rec['item_id'].apply(help)
    rec[['user_id', 'item_id']].drop_duplicates("user_id").to_csv('../result/result.csv', index=False, header=False)

def help( p ):
    ss = str(p).split(",")
    rec,viewed = ss[0],ss[1]
    rec = list( rec.split(" ") )

    size = 0
    for i in rec:
        size += 1
        if i in list( set( viewed.split(" ") ) ):
            rec.remove( i )
            size -= 1
        if size == 5:break

    rec = " ".join(rec[:5])
    return rec

def get_r4(  ):
    train = pd.read_csv('../data/train.csv')
    item_view = pd.read_csv('../data/filterItems.csv')
    users = pd.read_csv('../data/candidate.txt')

    #选择距end时间2小时内被view过的，其余的训练集item假定已经失去了时效，不再推荐
    end = train['action_time'].max()
    df = train[ train.action_time >= end-2*60*60 ][['user_id','item_id']]
    df = df.append( item_view )
    df = pd.merge( users,df,on='user_id' )
    df.groupby( ['item_id'],as_index=False ).count().sort_values( ['user_id'],ascending=False )

    recItems = " ".join( list(df['item_id'].head(10).values) )
    users['item_id'] = recItems
    rec = pd.merge(users,item_view,how='left', on='user_id').fillna("")
    rec['item_id'] = rec['item_id_x'] + "," + rec['item_id_y']
    rec['item_id'] = rec['item_id'].apply(help)
    users[['user_id', 'item_id']].drop_duplicates("user_id").to_csv('../result/result.csv', index=False, header=False)


if __name__ == "__main__":
    # r2()
    r3()




