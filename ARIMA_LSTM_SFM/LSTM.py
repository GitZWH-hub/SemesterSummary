#-*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from tensorflow.keras.layers import Dense, Dropout
import os
path = os.path.abspath(os.path.dirname(__file__))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i + look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def LSTM_model():
	# 固定np和tf的随机数种子，因为LSTM的参数初始化有随机性，这边固定即可
	np.random.seed(0)
	tf.random.set_seed(1234)
	# 1. 加载数据
	data = pd.read_csv(path + '/Data/DOW.csv')
	data = data.iloc[::-1]

	# 仅获取结算价
	close = data['Close'].values
	print("数据集长度:{}".format(len(data)))			# 2518
	# 2.
	# 变换形式：reshape(-1, 1)转换成仅1列的矩阵
	dataset = close.reshape(-1, 1)
	# 使用sklearn的MinMaxScaler进行数据归一化（转换成0到1之间的数值）
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	print("归一化后数据:{}".format(dataset))
	# 3. 划分数据集
	train_size = int(len(dataset) * 0.9)
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	print("训练集长度：{}".format(train_size))			# 2266

	# reshape into X=t and Y=t+1, timestep 步长200
	look_back = 200
	# 4. 重构成用于训练和测试的数据集
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	print('训练样本个数：', len(trainX))				# 2065

	# np.reshape(x, (batch_size , seq_len, input_dim))
	# 				 2065		  1		   1
	# 【batch_size】个样本同时训练；【seq_len】也就是LSTM中cell的个数；【input_dim】表示每个样本包含几个维度的数据（1维）
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	# 5. 创建模型、训练模型
	model = Sequential()
	# add了一个LSTM。其中256是输出维度；1是有多少个作为输入，look_back表示每次输入的一唯向量（步长）
	model.add(LSTM(256, input_shape=(1, look_back)))
	# dropout正则化：0~1之间的浮点数，控制输入线性变换的神经元断开比例；每层网络结点的舍弃率，防止过拟合
	model.add(Dropout(0.1))		
	model.add(Dense(1))
	# optimizer是优化损失函数方法adam
	model.compile(loss='mse', optimizer='adam')
	# 训练模型
		# epochs:训练次数 
		# batch_size:一次训练所选取的样本数
		# verbose:屏显模式0不输出 1输出进度 2输出每次训练结果    
	print(trainX)
	model.fit(trainX, trainY, epochs=1000, batch_size=2065, verbose=2)

	# 6. 测试集预测
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# 归一反转
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])

	# 7. 计算预测指标
	print('MSE: ', metrics.mean_squared_error(testY[0], testPredict[:,0]))
	print('MAE: ', metrics.median_absolute_error(testY[0], testPredict[:,0]))

	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

	# shift test predictions for plotting
	testPredictPlot = np.empty_like(dataset)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

	# plot baseline and predictions
	plt.plot(scaler.inverse_transform(dataset), label='Dataset')
	plt.plot(trainPredictPlot, label='Predict Trained')
	# testPrices = scaler.inverse_transform(dataset[test_size+look_back:])

	# 预测结果导出到文件
	# df = pd.DataFrame(data={"prediction": np.around(list(testPredict.reshape(-1)), decimals=2), "test_price": np.around(list(testPrices.reshape(-1)), decimals=2)})
	# df.to_csv("lstm_result.csv", sep=';', index=None)

	plt.plot(testPredictPlot, label='Predict Test')
	plt.legend(loc='best', fontsize=10) # 标签位置、大小
	plt.grid(True, ls=':', color='r', alpha=0.5)
	plt.title(u'LSTM')
	plt.show()
	a = 1

if __name__ == "__main__":
	LSTM_model()
