import os
import random

import gc
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split


def randomSelectTrain( train ):
    path = '../cache/train.pkl'
    if os.path.exists( path ):
        data = pickle.load(open(path, "rb"))
    else:
        # 抽样选择与用户阅读资讯相同数量的热门资讯
        data = pd.DataFrame()
        user_list = []
        item_list = []
        label_list = []
        items_pool = list(train['item_id'].values)
        for user, group in train.groupby(['user_id']):
            rec = dict()
            viewed_item = list(group['item_id'].unique() )
            for item in viewed_item:
                rec[item] = 1
                user_list.append(user)
                item_list.append(item)
                label_list.append(1)
            n = 0
            for i in range( 0, 3 * len(viewed_item) ):
                item = items_pool[random.randint(0, len(items_pool) - 1)]
                if item in rec.keys():
                    continue
                user_list.append(user)
                item_list.append(item)
                label_list.append(0)
                n += 1
                rec[item] = 0
                if n > len(viewed_item):
                    break
        data['user_id'] = user_list
        data['item_id'] = item_list
        data['label'] = label_list
        del user_list
        del item_list
        del label_list
        gc.collect()
        pickle.dump( data,open(path,'wb'),True )
    return data


def make_train_test(  ):
    train = pd.read_csv('../data/train.csv')
    data = randomSelectTrain( train )
    train,test = train_test_split(data,test_size=0.2,random_state=199009)
    train.to_csv( '../data/trainData.csv',index=False,header=False )
    test.to_csv('../data/testData.csv', index=False, header=False)

if __name__ == "__main__":
    # make_train_test()