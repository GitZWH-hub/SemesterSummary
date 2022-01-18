
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
from statsmodels.tsa.arima.model import ARIMA
import warnings
from sklearn import metrics
import os

path = os.path.abspath(os.path.dirname(__file__))
warnings.filterwarnings("ignore")

def ARIMA_PRE():
    # 1. 获取数据 —— CSV文件
    data = pd.read_csv(path + '/Data/DOW.csv')
    data = data.iloc[::-1]
    print(data.head())
    # 只保留时间和结算价
    df = data[['Date', 'Close']]
    df.Date = pd.to_datetime(df.Date)
    df = df.set_index('Date')

    # 2. 求ARIMA最合适的阶数
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))

    aic = []
    parameters = []
    for param in pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df, order=param, enforce_stationarity=True, enforce_invertibility=True)
            results = mod.fit()
            aic.append(results.aic)
            parameters.append(param)
        except:
            continue
    index_min = min(range(len(aic)), key=aic.__getitem__)
    print('the optimal model is: ARIMA {} - AIC {}'.format(parameters[index_min], aic[index_min]))
    
    # 训练集：取df总长度的 10分之9
    TRAIN = int(len(df) * 9 / 10)
    # 测试集：forecast剩余的 10分之2   
    TEST = len(df) - TRAIN

    # 3. 构建并训练模型
    model = ARIMA(df[:TRAIN], order=parameters[index_min])
    model_fit = model.fit()

    # 4. 模型预测
    # 求df在前TRAIN数据的起始截止日期
    pre_begin = (df.index[0]).strftime("%Y-%m-%d %H:%M:%S")[:7]
    pre_end =  (df.index[TRAIN-1]).strftime("%Y-%m-%d %H:%M:%S")[:10]
    # predict和forecast的区别：predict只能是训练集内部的数据，forecast可以预测外部数据
    # 这里直接predict200条已训练数据，并forecast了df剩下的数据
    pred = model_fit.predict(start=pre_begin, end=pre_end, typ='levels')
    pred2 = model_fit.forecast(TEST)

    index_key = df.index[TRAIN:]
    test = pd.Series(pred2.tolist(), index=index_key)
    # pred = pred.append(pd.Series(test))
    pred3 = pd.Series(test)

    # 5. 模型评估
    # 测试集真实值
    y_true = df.Close[TRAIN:]
    # 测试集预测值
    y_pred = pred2.tolist()
    print(y_true)
    # MSE
    print('MSE: ', (metrics.mean_squared_error(y_true, y_pred)))
    # MAE
    print('MAE: ', (metrics.mean_absolute_error(y_true, y_pred)))

    plt.figure(figsize=(8,6))
    plt.plot(df[:], label='Dataset')
    plt.plot(pred, label='Predict Trained')
    plt.plot(pred3, label='Predict Test')
    # plt.plot(df[TRAIN-1:], label='Real Data')
    plt.title(u'ARIMA')
    plt.legend(loc='best', fontsize=10) # 标签位置、大小
    plt.grid(True, ls=':', color='r', alpha=0.5)
    plt.show()


if __name__ == "__main__":

    # ARIMA本质上只能捕捉线性关系，而不能捕捉非线性关系。
    # 也就是说，采用ARIMA模型预测时序数据，必须是平稳的，如果不平稳的数据，是无法捕捉到规律的。
    # 股票数据是非稳定的，常常受政策和新闻的影响而波动，所以效果不佳，用一些谷歌搜索词变化提取一些特征，然后采用树模型预测可能会好点。

    # ARIMA 是用于单变量时间序列数据预测的最广泛使用方法之一
    # 优点：模型十分简单，只需要内生变量而不需要借助其他外生变量
    # 缺点：要求时序数据是稳定的；本质上只能捕捉线性关系，不能捕捉非线性关系；未考虑节假日以及特殊节日对预测结果的影响
    ARIMA_PRE()

    # https://www.relataly.com/stock-market-prediction-using-multivariate-time-series-in-python/1815/
    # 经济周期、政治发展、不可预见的事件、心理因素、市场情绪，甚至天气，所有这些变量或多或少都会对价格产生影响。
    # 此外，这些变量中有许多是相互依赖的，这使得统计建模更加复杂。多元神经网络
    # （1）选择股票数据中已经存在的数据作为特征，如开仓价、最高价、最低价等等
    # （2）爬取新闻等获取政策等影响因素作为特征
