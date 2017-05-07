# coding=utf-8
import pandas as pd

def get_viewed_item():
    test = open('../data/test.txt','rb')
    csv = pd.DataFrame()
    while True:
        line = test.readline()
        if not line:
            break
        tmpcsv = pd.DataFrame()
        line = line.split('\r\n')[0]
        uid,items = line.split(",")[0],line.split(",")[1]
        items = items.split(" ")
        tmpcsv['user_id'] = [uid] * len(items)
        # print tmpcsv
        tmpcsv['item_id'] = items
        csv = csv.append( tmpcsv )
    csv.to_csv('../data/filterItems.csv',index=None)

def get_rating_matrix(  ):
    train = pd.read_csv('../data/train.csv')
    news = pd.read_csv('../data/news_info.csv')

    train = pd.merge(train, news, on='item_id')
    train = train[['user_id','item_id','action_type']]
    train['weight'] = train['action_type'].apply(lambda p:1 if p in ['view' 'deep_view'] else 5)
    train = train[['user_id','item_id','weight']].groupby( ['user_id','item_id'],as_index=False ).sum()
    #归一化,5分制
    train['weight'] = ( train['weight'] - train['weight'].min() ) / ( train['weight'].max() - train['weight'].min() )
    return train

def get_item_matrix( ):
    ratrings = get_rating_matrix()
    ratrings = ratrings.sort_values(['user_id','item_id']).head(100000)
    item_matrxi = pd.pivot_table( ratrings,values='weight',index='user_id',columns='item_id',fill_value=0)
    return item_matrxi

if __name__ == "__main__":
    # ratrings = get_rating_matrix()
    # ratrings.to_csv('../data/rating.csv',index=None)

    mat = get_item_matrix()
    mat.to_csv('../data/rate_mat.csv',index=None)
    # print mat.head()