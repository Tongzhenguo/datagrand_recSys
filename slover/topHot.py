# coding=utf-8
"""
    为用户推荐最热的咨询，如果用户以前看过不会再推荐
    物品流行度公式：参考hacker news排序算法，还有一个是《推荐系统实践》中给出的排序算法
"""
from math import log
import pandas as pd
import time


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

def TopItemsInCate( ):
    print('在n+1天中出现过的cate,取最热门的top 10')
    te = pd.read_csv('../data/test.csv')
    item = pd.read_csv('../data/news_info.csv')
    res = pd.merge(item[['item_id', 'cate_id']], te[['item_id']],on='item_id')
    cate_items_dict = {}
    for cate_id,group in res.groupby(['cate_id'],as_index=False):
        cate_items_dict[ cate_id ] = list(group.groupby( ['item_id'],as_index=False ).count().sort_values( ['cate_id'],ascending=False )['item_id'].head(10).values)

    print('全品类最热')
    cate_items_dict['all_cate'] = list(
        res.groupby(['item_id'], as_index=False).count().sort_values(['cate_id'], ascending=False)['item_id'].head(
            10).values)
    cate_items_dict['3_13'] = ['532349','503663','529959','557579','558082','558788','557167','558910']

    return cate_items_dict

def TopItemsInCateAndTime( ):
    te = pd.read_csv('../data/test.csv')
    ninfo = pd.read_csv('../data/news_info.csv')
    item_action_cnt = te.groupby(['item_id'], as_index=False).count()
    item_action_cnt.columns = ['item_id', 'action_cnt']

    item_pop = pd.merge(ninfo, te, on='item_id')
    item_pop = pd.merge(item_action_cnt, item_pop, on='item_id')
    end = time.mktime(time.strptime('2017-2-19 23:59:59', '%Y-%m-%d %H:%M:%S'))
    item_pop['pop'] = item_pop['action_cnt'] / pow((end - item_pop['timestamp']) / 3600, 1.8)
    item_pop = item_pop.groupby(['cate_id','item_id'], as_index=False).sum()[['cate_id','item_id','pop']]
    item_pop = item_pop.sort_values(['pop'], ascending=False)

    cate_items_dict = {}
    for cate_id, group in item_pop.groupby(['cate_id'], as_index=False):
        cate_items_dict[cate_id] = list(group['item_id'].head(10).values)

    print('全品类最热')
    cate_items_dict['all_cate'] = list(item_pop['item_id'].head(10).values)
    cate_items_dict['3_13'] = ['532349', '503663', '529959', '557579', '558082', '558788', '557167', '558910']
    cate_items_dict['3_7'] = ['565225', '532502', '522941', '557579', '558082', '558788', '557167', '558910']
    cate_items_dict['1_19'] = ['554018','548664','542740','455410','557579', '558082', '558788', '557167', '558910']
    #test中没有出现过，取发布时间戳最晚的
    cate_items_dict['3_20'] = list( ninfo[ninfo.cate_id=='3_20'].sort_values(['timestamp'],ascending=False).head(10)['item_id'].values )
    cate_items_dict['3_15'] = list(ninfo[ninfo.cate_id == '3_15'].sort_values(['timestamp'], ascending=False).head(10)['item_id'].values)
    return cate_items_dict

def hourtTomorOReve( tm_hour ):
    if( tm_hour in [6,7,8,9] ):return 'mor'
    elif( tm_hour in [19,20,21,22] ):return 'eve'
    else:return ''

#用户是早晨看资讯多，还是晚上看资讯多
def get_user_morOReve(  ):
    tr = pd.read_csv('../data/train.csv')
    tr['morOReve'] = tr['action_time'].apply( lambda t: hourtTomorOReve(time.localtime(t).tm_hour)  )
    tr = tr[ tr['morOReve']!='' ][['user_id','morOReve','action_time']].groupby( ['user_id','morOReve'],as_index=False ).count()
    tr = tr.sort_values( ['action_time'],ascending=False )[['user_id','morOReve']].drop_duplicates()
    return tr

#资讯在早晨阅读多，还是晚上阅读多
def get_item_morOReve(  ):
    tr = pd.read_csv('../data/train.csv')
    tr['morOReve'] = tr['action_time'].apply(lambda t: hourtTomorOReve(time.localtime(t).tm_hour))
    tr = tr[tr['morOReve'] != ''][['item_id', 'morOReve', 'action_time']].groupby(['item_id', 'morOReve'],
                                                                                  as_index=False).count()
    tr = tr.sort_values(['action_time'], ascending=False)[['item_id', 'morOReve']].drop_duplicates()
    return tr

def userPrefCate(  ):

    cate_item_d = TopItemsInCateAndTime()
    tr = pd.read_csv('../data/train.csv')
    rec = pd.read_csv('../data/candidate.txt')
    users = pd.read_csv('../data/candidate.txt')
    test = pd.read_csv('../data/test.csv')
    user_pref = pd.DataFrame()
    user_list = []
    item_list = []
    for u,group in tr.groupby( ['user_id'],as_index=False ):
        sortedG = group.groupby(['cate_id'], as_index=False).count().sort_values(['item_id'], ascending=False)
        cate_id__head = sortedG['cate_id'].head(3)
        u_view_count = len(group)
        cate_view_ratio = sortedG
        cate_view_ratio['ratio'] = cate_view_ratio['item_id'].apply( lambda c:float(c)/u_view_count  )
        cate_view_ratio.index = cate_view_ratio.cate_id
        cate_view_ratio_d = cate_view_ratio['ratio'].to_dict()
        user_list.append(u)
        if cate_view_ratio_d[ cate_id__head.head(1).values[0] ]>=0.4:#兴趣集中：只关注一种cate
            user_pref_cate = cate_id__head.head(1).values[0]
            print user_pref_cate
            item_list.append(' '.join(map(str, cate_item_d[user_pref_cate])))
        #兴趣广泛，推全局最火的
        elif len(list( cate_id__head.values ))>1:
            item_list.append(' '.join(map(str, cate_item_d['all_cate'])))
        else:#兴趣集中：只关注一种cate
            user_pref_cate = cate_id__head.head(1).values[0]
            print user_pref_cate
            item_list.append( ' '.join( map(str,cate_item_d[user_pref_cate]) ) )
    user_pref['user_id'] = user_list
    user_pref['item_id'] = item_list
    rec = pd.merge(user_pref, rec, on='user_id')

    print('过滤掉用户已经看过的')
    user_list = []
    viewed_list = []
    for user, group in tr[['user_id', 'item_id']].groupby(['user_id']):
        user_list.append(user)
        viewed_list.append(" ".join(map( str,list(group['item_id'].values) )))
    user_viewed = pd.DataFrame()
    user_viewed['user_id'] = user_list
    user_viewed['item_id'] = viewed_list

    rec = pd.merge(rec, user_viewed, how='left', on='user_id').fillna("")  # item_view
    rec['item_id'] = rec['item_id_x'] + "," + rec['item_id_y']
    rec['item_id'] = rec['item_id'].apply(help)
    rec = rec[['user_id', 'item_id']]

    print('还有部分用户没关注过物品候选集,推荐test中的topHot5')
    topHot = \
        test.groupby(['item_id'], as_index=False).count().sort_values(['user_id'], ascending=False).head(5)[
            'item_id'].values
    oldrec = users
    oldrec['oldrec_item'] = [" ".join(map(str, list(topHot)))] * len(oldrec)
    oldrec = pd.merge(oldrec, rec, how='left', on='user_id', ).fillna(0)
    oldrec = oldrec[oldrec.item_id == 0][['user_id', 'oldrec_item']]
    oldrec.columns = ['user_id', 'item_id']
    rec = rec.append(oldrec)

    rec.drop_duplicates('user_id').to_csv('../result/result.csv', index=None, header=None)


if __name__ == "__main__":
    userPrefCate()
