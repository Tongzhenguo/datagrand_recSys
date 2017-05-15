import pandas as pd

def make_train_set( train,user ):
    train = pd.merge( user,train )
    train['rating'] = 1
    pos = train[['rating','user_id','item_id']].drop_duplicates()

    #选择最火的300个资讯做负类
    hot300 = train.groupby(['item_id'],as_index=False)[['user_id','item_id']].count().head(300)['item_id']
    neg = pd.DataFrame()
