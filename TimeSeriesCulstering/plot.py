import pandas as pd
import sqlite3 as sql3
import matplotlib.pyplot as plt
import os

path = os.path.abspath(os.path.dirname(__file__))
DBNAME = path + '/DataBase/HiDBData'
conn = sql3.connect(DBNAME)

qry = 'select *from CU where ts_code like \'CU1801.SHF\''
data1 = pd.read_sql_query(qry, conn)
df1 = data1[['trade_date', 'close']]
df1.trade_date = pd.to_datetime(df1.trade_date)
df1 = df1.set_index('trade_date')

qry = 'select *from CU where ts_code = \'CU1802.SHF\''
data2 = pd.read_sql_query(qry, conn)
df2 = data2[['trade_date', 'close']]
df2.trade_date = pd.to_datetime(df2.trade_date)
df2 = df2.set_index('trade_date')

# qry = 'select *from CU where ts_code = \'CU1803.SHF\''
# data3 = pd.read_sql_query(qry, conn)
# df3 = data3[['trade_date', 'close']]
# df3.trade_date = pd.to_datetime(df3.trade_date)
# df3 = df3.set_index('trade_date')

# 时间序列画图对比
fig= plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(211)
ax1.plot(df1)
ax2 = fig.add_subplot(212)
ax2.plot(df2)
# ax3 = fig.add_subplot(313)
# ax3.plot(df3)
plt.show()