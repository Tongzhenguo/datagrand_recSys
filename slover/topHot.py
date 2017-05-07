# coding=utf-8
"""
    为用户推荐最热的咨询，如果用户以前看过不会再推荐
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

def help( p ):
    ss = str(p).split(",")
    if len( ss ) < 2:return " ".join( ss[0].split(" ")[:5] )
    rec,viewed = ss[0],ss[1]
    rec = list( set(rec.split(" ")) - set( viewed.split(" ") ) )[:5]
    rec = " ".join(rec)
    return rec


if __name__ == "__main__":
    r2()





