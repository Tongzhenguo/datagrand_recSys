# coding=utf-8
"""
尝试不同相似系数
pandas 实现
"""
import math
from itertools import permutations
import pandas as pd

def get_concur_mat():
    rat = pd.read_csv('../data/rating.csv')

    sim_mat = pd.DataFrame()
    item1_list = []
    item2_list = []
    concur_count = []
    user_groups = rat.groupby( ['user_id'] )
    for name,group in user_groups:
        for pair in permutations(list(group['item_id'].values), 2):
            item1_list.append( pair[0] )
            item2_list.append( pair[1] )
            concur_count.append( 1 )
        # print name
    sim_mat['item1'] = item1_list
    sim_mat['item2'] = item2_list
    sim_mat['count'] = concur_count
    sim_mat.to_csv('../data/sim/concurrence_mat.csv',index=False)

def entropy( *elements ):
    sum = 0
    for element in elements:
        sum += element
    result = 0.0
    for x in elements:
        zeroFlag = 1 if x == 0 else 0
        result += x * math.log((x + zeroFlag) / sum)
    return -result
"""
        k11：事件A与事件B同时发生的次数
　　　　k12：B事件发生，A事件未发生
　　　　k21：A事件发生，B事件未发生
　　　　k22：事件A和事件B都未发生
        double rowEntropy = entropy(k11, k12) + entropy(k21, k22);
        double columnEntropy = entropy(k11, k21) + entropy(k12, k22);
        double matrixEntropy = entropy(k11, k12, k21, k22);
        return 2 * (matrixEntropy - rowEntropy - columnEntropy);
"""
def get_loglike( ):
    concur_mat = pd.read_csv('../data/sim/concurrence_mat.csv').groupby(['item1', 'item2'], as_index=False).sum()
    item_vector = pd.read_csv('../data/rating.csv')[['item_id', 'user_id']].groupby(['item_id'], as_index=False).count()
    item_vector.index = item_vector['item_id']
    item_vector.columns = ['item_id', 'count']
    item_count_dict = item_vector['count'].to_dict()
    concur_mat['item1_count'] = concur_mat['item1'].apply(lambda p: item_count_dict[p])
    concur_mat['item2_count'] = concur_mat['item2'].apply(lambda p: item_count_dict[p])
    rowEntropy = H(  )

    concur_mat['sim'] = concur_mat['count'] / ( concur_mat['item1_count'].apply(math.sqrt) * concur_mat['item2_count'].apply(math.sqrt))

def get_concur_sim(  ):
    concur_mat = pd.read_csv( '../data/sim/concurrence_mat.csv' ).groupby( ['item1','item2'],as_index=False).sum()
    item_vector = pd.read_csv( '../data/rating.csv' )[['item_id','user_id']].groupby(['item_id'],as_index=False).count()
    item_vector.index = item_vector['item_id']
    item_vector.columns = ['item_id','count']
    item_count_dict = item_vector['count'].to_dict()
    concur_mat['item1_count'] = concur_mat['item1'].apply( lambda p:item_count_dict[p] )
    concur_mat['item2_count'] = concur_mat['item2'].apply(lambda p: item_count_dict[p])
    concur_mat['sim'] = concur_mat['count'] / ( concur_mat['item1_count'].apply(math.sqrt)  * concur_mat['item2_count'].apply(math.sqrt) )

    sim_mat = pd.DataFrame()
    for item1,group in concur_mat.groupby( ['item1'],as_index=False ):
        df = group.sort_values( ['sim'],ascending=False ).head( 10 )
        df['item1'] = [item1] * len(df)
        # print('------------------------------')
        sim_mat = sim_mat.append( df )
    sim_mat[['item1','item2','sim']].to_csv( '../data/sim/concurrence_sim.csv',index=False )

if __name__ == "__main__":
    get_concur_mat()
    get_concur_sim()

