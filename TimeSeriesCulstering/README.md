# 期货合约时间序列聚类


- **FutureClustering.py**
1. 上期所、大商所、郑商所共35个品种的时间序列聚类。结果如图所示：
![Zhang Weihua](https://github.com/GitZWH-hub/SemesterSummary/blob/main/TimeSeriesCulstering/PNG/FutureClustering.png)
![Zhang Weihua](https://github.com/GitZWH-hub/SemesterSummary/blob/main/TimeSeriesCulstering/PNG/result.png)
2. 参考sklearn官方链接:  [sklearn](https://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html?highlight=plot%20stock%20market#sphx-glr-download-auto-examples-applications-plot-stock-market-py)

- **HierarchicalClustering.py**
1. 层次聚类
2. 数据集：同一CU品种、不同截止日期的时间序列之间的聚类，如（CU2101、CU2102、CU1910）
3. 结果：相邻截止日期的合约最先聚，如下图的CU1801、CU1802的曲线：
![Zhang Weihua](https://github.com/GitZWH-hub/SemesterSummary/blob/main/TimeSeriesCulstering/PNG/Futures.png)
4. 时间序列间相似度矩阵持久化到csv文件中