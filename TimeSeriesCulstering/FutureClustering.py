import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
from sklearn import cluster, covariance, manifold
import sqlite3 as sql3
import os

plt.rcParams['font.sans-serif']=['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
quotes = []
date = '1801'


'''
上期所: SHFE
'''
print(" ++ read DBData data ++ ")
symbol_dict = {
    # "SP": "纸浆",
    # "WR": "线材",
    # "AU": "黄金",
    # "BU": "石油沥青",
    # "FU": "燃料油",

    "AG": "沪银",
    "AL": "沪铝",
    "CU": "沪铜",
    "HC": "热压卷板",
    "NI": "沪镍",
    "PB": "沪铅",
    "RB": "螺纹钢",
    "RU": "橡胶",
    "SN": "沪锡",
    "ZN": "沪锌"
}
symbols, nameA = np.array(sorted(symbol_dict.items())).T
path = os.path.abspath(os.path.dirname(__file__))
DBNAME = path + '/DataBase/DBData'
conn = sql3.connect(DBNAME)

for sysmbol in symbols:
    qry = 'select trade_date, open, close from {} where ts_code like \'%{}%\''
    data = pd.read_sql_query(qry.format(sysmbol, sysmbol + date), conn)
    # 缺失值填充mean
    data.open.fillna(data['open'].mean(), inplace=True)
    data.close.fillna(data['close'].mean(), inplace=True)
    print(sysmbol, len(data))
    if len(data) == 244:
        quotes.append(data)
conn.close()

'''
大商所: DCE
'''
print(" ++ read DBDataDCM data ++ ")
sysmbol_dict_DCE = {
    # "A": "豆一",
    # "EG": "乙二醇",
    # "BB": "胶合板",
    # "JD": "鸡蛋",

    "B": "豆二",
    "C": "玉米",
    "CS": "玉米淀粉",
    "FB": "纤维板",
    "I": "铁矿石",
    "J": "焦炭",
    "JM": "焦煤",
    "L": "塑料",
    "M": "豆粕",
    "P": "棕榈油",
    "PP": "聚丙烯",
    "V": "PVC聚氯乙烯",
    "Y": "豆油"
}

symbols, nameB = np.array(sorted(sysmbol_dict_DCE.items())).T
DBNAME = path + '/DataBase/DBDataDCE'
conn = sql3.connect(DBNAME)
for sysmbol in symbols:
    qry = 'select trade_date, open, close from {} where ts_code like \'%{}%\''
    data = pd.read_sql_query(qry.format(sysmbol, sysmbol + date), conn)
    data.open.fillna(data['open'].mean(), inplace=True)
    data.close.fillna(data['close'].mean(), inplace=True)
    print(sysmbol, len(data))
    if len(data) == 244:
        quotes.append(data)
conn.close()

'''
郑商所: CZCE
'''
print(" ++ read DBDataCZCE data ++ ")
sysmbol_dict_CZCE = {
    # "AP": "苹果",
    # "CJ": "红枣",
    # "CY": "棉纱",
    # "PM": "普麦",
    # "RS": "菜籽",
    # "SR": "白糖",

    "LR": "晚籼粳",
    "RM": "菜粕",
    "CF": "郑棉",
    "FG": "玻璃",
    "JR": "粳稻",
    "MA": "甲醇",
    "OI": "菜油",
    "RI": "早籼稻",
    "SF": "硅铁",
    "SM": "锰硅",
    "TA": "PTA精对苯二甲酸",    # 生产聚酯纤维
    "WH": "强麦",
    "ZC": "动力煤"
}
symbols, nameC = np.array(sorted(sysmbol_dict_CZCE.items())).T
DBNAME = path + '/DataBase/DBDataCZCE'
conn = sql3.connect(DBNAME)
for sysmbol in symbols:
    qry = 'select trade_date, open, close from {} where ts_code like \'%{}%\''
    data = pd.read_sql_query(qry.format(sysmbol, sysmbol + date), conn)
    data.open.fillna(data['open'].mean(), inplace=True)
    data.close.fillna(data['close'].mean(), inplace=True)
    print(sysmbol, len(data))
    if len(data) == 244:
        quotes.append(data)
conn.close()

# 到这里为止，每行是一个品种时间序列
close_prices = np.vstack([q["close"] for q in quotes])
open_prices = np.vstack([q["open"] for q in quotes])

variation = (close_prices - open_prices)/ open_prices * 100
# 到这里为止，每行是一个品种的时间序列（处理后的价格）

'''(1)'''
# 学习一个图结构
edge_model = covariance.GraphicalLassoCV()

# 我们使用稀疏逆协方差估计来找出哪些报价是有条件地与其他报价相关的。 
# 具体来说，稀疏逆协方差为我们提供了一个图，即连接列表。 对于每个符号，它所连接的符号也有助于解释其波动。
X = variation.copy().T
# 到这里为止，转置了，每列是一个品种的时间序列
X /= X.std(axis=0)      # x.std(axis=0)计算每一列的标准差，除以标准差就变为等值单位（忽略掉单位问题）
edge_model.fit(X)       # fit是为了求得不同列（不同品种的之间的协方差矩阵，协方差矩阵的主对角线实际上是方差）

print("协方差矩阵")
print(edge_model.covariance_)
'''(2)'''
# 使用AP聚类 啥是AP聚类
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
n_labels = labels.max()

names = np.append(nameA, nameB)
names = np.append(names, nameC)
for i in range(n_labels + 1):
    print("Cluster %i: %s" % ((i + 1), ", ".join(names[labels == i])))

node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver="dense", n_neighbors=6
)

embedding = node_position_model.fit_transform(X.T).T


plt.figure(1, facecolor="w", figsize=(10, 8))
plt.clf()
ax = plt.axes([0.0, 0.0, 1.0, 1.0])
plt.axis("off")


partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = np.abs(np.triu(partial_correlations, k=1)) > 0.02


plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels, cmap=plt.cm.nipy_spectral)
start_idx, end_idx = np.where(non_zero)

segments = [
    [embedding[:, start], embedding[:, stop]] for start, stop in zip(start_idx, end_idx)
]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(
    segments, zorder=0, cmap=plt.cm.hot_r, norm=plt.Normalize(0, 0.7 * values.max())
)
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

for index, (name, label, (x, y)) in enumerate(zip(names, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = "left"
        x = x + 0.002
    else:
        horizontalalignment = "right"
        x = x - 0.002
    if this_dy > 0:
        verticalalignment = "bottom"
        y = y + 0.002
    else:
        verticalalignment = "top"
        y = y - 0.002
    plt.text(
        x,
        y,
        name,
        size=10,
        horizontalalignment=horizontalalignment,
        verticalalignment=verticalalignment,
        bbox=dict(
            facecolor="w",
            edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
            alpha=0.6,
        ),
    )

plt.xlim(
    embedding[0].min() - 0.15 * embedding[0].ptp(),
    embedding[0].max() + 0.10 * embedding[0].ptp(),
)
plt.ylim(
    embedding[1].min() - 0.03 * embedding[1].ptp(),
    embedding[1].max() + 0.03 * embedding[1].ptp(),
)

plt.show()

