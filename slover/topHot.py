# coding=utf-8
"""
    为用户推荐最热的咨询，如果用户以前看过不会再推荐
"""
import pandas as pd
import time


def r1():
    print('topHot 推荐')
    train = pd.read_csv('../data/train.csv')
    train['item_id'] = train['item_id'].apply(str)
    rec = pd.read_csv('../data/candidate.txt')

    end = time.mktime(time.strptime('2017-2-18 12:00:00', '%Y-%m-%d %H:%M:%S'))

    train = pd.merge( rec,train[train['action_time']>=end],on='user_id' )
    hot_item = train[['user_id', 'item_id']].groupby(['item_id'], as_index=False).count().sort_values(['user_id'],ascending=False)[['item_id']]
    rec['item_id'] = " ".join( list(hot_item['item_id'].values)[:20] )

    print('过滤掉用户已经看过的')
    user_list = []
    viewed_list = []
    for user, group in train[['user_id', 'item_id']].groupby(['user_id']):
        user_list.append(user)
        viewed_list.append(" ".join(list(group['item_id'].values)))
    user_viewed = pd.DataFrame()
    user_viewed['user_id'] = user_list
    user_viewed['item_id'] = viewed_list

    rec = pd.merge(rec, user_viewed, how='left', on='user_id').fillna("")  # item_view
    rec['item_id'] = rec['item_id_x'] + "," + rec['item_id_y']
    rec['item_id'] = rec['item_id'].apply(help)
    rec = rec[['user_id', 'item_id']]

    rec.to_csv('../result/result.csv',index=False,header=False)

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

if __name__ == "__main__":
    r1()




