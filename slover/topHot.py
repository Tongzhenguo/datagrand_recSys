# coding=utf-8
"""
    为用户推荐最热的咨询，如果用户以前看过不会再推荐
    物品流行度公式：分别参考了rabbit，hacker news两个网站额排序算法，还有一个是《推荐系统实践》中给出的排序算法
"""
from math import log

import pandas as pd
import time


def rabbit_rank( s,seconds, ):
        order = log(max(abs(s), 1), 10)
        sign = 1 if s > 0 else -1 if s < 0 else 0
        return round(order + sign * seconds / 45000, 7)

def hacker_news_rank(  ):
    #参考自http://www.oschina.net/news/43456/how-hacker-news-ranking-algorithm-works
    tr = pd.read_csv('../data/train.csv')
    item = pd.read_csv('../data/news_info.csv')
    item_action_cnt = tr[['user_id','item_id','action_type']].drop_duplicates().groupby(['item_id'],as_index=False).count()[['item_id','action_type']]
    item_action_cnt.columns = ['item_id','action_cnt']
    item_pop = pd.merge(item[['item_id', 'timestamp']], tr, on='item_id')
    item_pop = pd.merge( item_action_cnt,item_pop,on='item_id' )
    item_pop['pop'] = item_pop['action_cnt'] / pow( ( item_pop['action_time'] - item_pop['timestamp'] )/3600 ,5.8 ) #5.8等于10.8，优于1.8,2.8
    item_pop = item_pop[['item_id','pop']].groupby( ['item_id'],as_index=False ).sum()
    hot_item = item_pop.sort_values(['pop'], ascending=False)
    rec = pd.read_csv('../data/candidate.txt')
    rec['item_id'] = " ".join(list(hot_item['item_id'].apply(str).values)[:20])

    print('过滤掉用户已经看过的')
    user_list = []
    viewed_list = []
    for user, group in tr[['user_id', 'item_id']].groupby(['user_id']):
        user_list.append(user)
        viewed_list.append(" ".join(list(group['item_id'].apply(str).values)))
    user_viewed = pd.DataFrame()
    user_viewed['user_id'] = user_list
    user_viewed['item_id'] = viewed_list

    rec = pd.merge(rec, user_viewed, how='left', on='user_id').fillna("")  # item_view
    rec['item_id'] = rec['item_id_x'] + "," + rec['item_id_y']
    rec['item_id'] = rec['item_id'].apply(help)
    rec = rec[['user_id', 'item_id']]

    rec.to_csv('../result/result.csv', index=False, header=False)



def top_pop(  ):
    #参考自推荐系统时间p130
    tr = pd.read_csv('../data/train.csv')
    tr['pop'] = tr['action_time'].apply(lambda t: 1 / (1.0 + 0.2 * (1487433599 - t))) #0.2优于0.1和0.5
    item_pop = tr[['item_id', 'pop']].groupby(['item_id'], as_index=False).sum()
    hot_item =  item_pop.sort_values( ['pop'],ascending=False )

    rec = pd.read_csv('../data/candidate.txt')
    rec['item_id'] = " ".join(list(hot_item['item_id'].apply(str).values)[:20])

    print('过滤掉用户已经看过的')
    user_list = []
    viewed_list = []
    for user, group in tr[['user_id', 'item_id']].groupby(['user_id']):
        user_list.append(user)
        viewed_list.append(" ".join(list(group['item_id'].apply(str).values)))
    user_viewed = pd.DataFrame()
    user_viewed['user_id'] = user_list
    user_viewed['item_id'] = viewed_list

    rec = pd.merge(rec, user_viewed, how='left', on='user_id').fillna("")  # item_view
    rec['item_id'] = rec['item_id_x'] + "," + rec['item_id_y']
    rec['item_id'] = rec['item_id'].apply(help)
    rec = rec[['user_id', 'item_id']]

    rec.to_csv('../result/result.csv', index=False, header=False)

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
    # top_pop()
    hacker_news_rank()



