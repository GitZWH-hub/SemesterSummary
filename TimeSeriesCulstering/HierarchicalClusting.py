import sqlite3 as sql3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, dendrogram
import os
plt.rcParams['font.sans-serif']=['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


def TimeSeriesSimilarity(s1, s2):
    l1 = len(s1)
    l2 = len(s2)
    paths = np.full((l1+1, l2+1), np.inf)
    paths[0,0] = 0
    for i in range(l1):
        for j in range(l2):
            d = s1[i] - s2[j]
            cost = d ** 2
            paths[i+1, j+1] = cost + min(paths[i, j+1], paths[i+1, j], paths[i, j])
    
    paths = np.sqrt(paths)
    s = paths[l1, l2]
    return s, paths.T

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--window', type=int, default=10)
    parser.add_argument('-n', '--nums', type=int, default=20)
    args = parser.parse_args()

    '''（1）首先获取所有的时间序列'''
    path = os.path.abspath(os.path.dirname(__file__))
    DBNAME = path + '/DataBase/HiDBData'
    conn = sql3.connect(DBNAME)

    qry = 'select *from CU'
    data = pd.read_sql_query(qry, conn)
    # data存放了所有时间序列的数据
    ts_code_column_uniques = data['ts_code'].unique()   # 把tscode列取出做一个array
    df_dict = {}
    for ts_code_column in tqdm(ts_code_column_uniques):
        df_temp = data[data['ts_code'].isin([ts_code_column])]
        # 只保留两列
        df_temp = df_temp[['trade_date', 'close']]
        df_temp.trade_date = pd.to_datetime(df_temp.trade_date)
        df_temp.set_index('trade_date')
        df_dict[ts_code_column] = df_temp

    '''（2）时间序列预处理模块'''
    for ts_code in tqdm(df_dict):
        # 缺失值处理,填充为当前序列的平均值
        df_dict[ts_code].close.fillna(df_dict[ts_code]['close'].mean(), inplace=True)
        # reverse
        df_dict[ts_code] = df_dict[ts_code].iloc[::-1]

    '''（3）时间序列基线提取'''
    # baseline_dict = {}
    # for ts_code in tqdm(df_dict):
        # 移动平均算法提取时间序列基线，窗口大小w
        # 存在疑问，是否就是arma的训练拟合线，感觉不像

    '''（4）'''
    # 有了基线之后，可以从基线中提取时间序列特征然后利用常用的knn等进行分类。
    # 或者使用kmeans等方法来聚类(使用时间序列相似度来聚类(层次聚类)，具体使用动态规划的dtw)
    keys = list(df_dict.keys())[:args.nums]
    length = len(keys)
    dist = np.zeros((length, length))
    for i in tqdm(range(0, length)):
        for j in range(i + 1, length):
            distance, paths = TimeSeriesSimilarity(df_dict[keys[i]].close.tolist(), df_dict[keys[j]].close.tolist())
            dist[i][j] = dist[j][i] = round(distance, 2)
            

    # # 以pandas.dataframe保存到文件中
    df = pd.DataFrame(dist)

    csv_path = path + '/csv/' + str(length) + '_len.csv'
    df.to_csv(csv_path)

    df = pd.read_csv(csv_path)

    results_clusters = linkage(df.values, method='complete', metric='euclidean')
    clusters = pd.DataFrame(results_clusters, columns=['label1', 'label2', 'distance', 'sampleSize'],
                            index=['clusters %d'%(i+1) for i in range(results_clusters.shape[0])])

    clusters.to_csv(path + '/csv/result.csv')

    dendr = dendrogram(results_clusters)
    plt.show()
    b = 1
    # 聚类结果如何保存?
    for index, row in clusters.iterrows():
        # keys组合
        keys.append([keys[int(row['label1'])], keys[int(row['label2'])]])
        print([keys[int(row['label1'])], keys[int(row['label2'])]])





