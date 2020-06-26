import keras
import numpy as np
import matplotlib.pyplot as plt
# Sequential按顺序构成的模型
from keras.models import Sequential
# Dense全连接层
from keras.layers import Dense
# 使用numpy生成100个随机点
x_data = np.random.rand(100)
noise = np.random.normal(0,0.01,x_data.shape)
y_data = x_data * 0.1 + 0.2 + noise
# 显示随机点

plt.scatter(x_data,y_data)
plt.show()
# 构建一个顺序模型
model = Sequential()
# 在模型中添加一个全连接层
# 输出为1
# 输入的数据为1维度
model.add(Dense(units=1,input_dim=1))
# sgd:Stochastic gradient descent 随机梯度下降法(求出w,b)的参数
# mse:Mean Squared Error 均方误差 (真实值和预测值的差)越小越好
model.compile(optimizer='sgd',loss='mse')
# 训练3001个批次
for step in range(3001):
    # 每次训练一个批次
    cost = model.train_on_batch(x_data,y_data)
    # 每500个batch打印一次cost值
    if step * 500 == 0:
        print('cost',cost)
# 打印权值和bias值
# 因为是只有第一层所以写成[0]
W,b = model.layers[0].get_weights()
print('W',W,'b',b)
# 输入一维数据,模型预测
y_pred = model.predict(x_data)
# 显示随机点
plt.scatter(x_data,y_data)
# 显示红色线的预测结果
plt.plot(x_data,y_pred,'r-',lw=3)
plt.show()