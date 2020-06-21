"""
机器学习 -- 线性回归
预测第十小时的 pm2.5 值 by using adagram
"""

import numpy as np
import pandas as pd
import math
import csv

# 总结 可以在loss_function 中添加正则项
# 超参数的调节： b = 1， 初始w 可以再变化  模型可以更优秀

data_init = pd.read_csv("/Users/jingrancao/Downloads/数据/hw1/train.csv", header=0, encoding="big5")
data = data_init.copy()

data.replace(to_replace="NR", value=0, inplace=True)
data = data.iloc[:, 3:]
data_np = data.to_numpy()

month_data = {}
# 分割数据 滑动窗口的思想
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day*24: (day+1)*24] = data_np[18*(20*month+day): 18*(20*month+day+1), :]

    month_data[month] = sample

# 特征提取 一个label 9个小时 * 18 个特征
x_train = np.empty([12*471, 18*9], dtype=np.float64)
y_train = np.empty([12*471, 1], dtype=np.float64)

for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x_train[month*471+day*24+hour, :] = month_data[month][:, day*24 + hour: day*24 + hour + 9].reshape(1, -1)
            y_train[month*471+day*24+hour] = month_data[month][9, day*24+hour+9]

# 数据的标准化
x_train_mean = np.mean(x_train, axis=0)  # 0 -> 列
x_train_std = np.std(x_train, axis=0)

for i in range(len(x_train)):
    for j in range(len(x_train[0])):
        if x_train_std[j] == 0:
            x_train_std[j] = 0.0000001
        x_train[i][j] = (x_train[i][j] - x_train_mean[j]) / x_train_std[j]


# processing train_test_split
# 由于训练集没有给 标签 所以需要用 train_data 测一下
x_train_set = x_train[:math.floor(len(x_train)*0.8), :]
x_test_set = x_train[math.floor(len(x_train)*0.8):, :]
y_train_set = y_train[:math.floor(len(y_train)*0.8)]
y_test_set = y_train[math.floor(len(y_train)*0.8):]


# model_training by using loss function and gradient decent - learning rate - adagram
dim = 18*9 + 1  # 加1 是为了 对bias做更好的处理
w = np.ones([dim, 1])

x_train = np.concatenate((np.ones([12*471, 1]), x_train), axis=1)   # 1 是列
learning_rate = 100
iter_time = 10000  # 迭代次数
adagram = np.zeros([dim, 1])  # learning_rate/sqrt(sum_of_pre_grads**2)
eps = 0.000000001  # 避免 adagram=0

for time in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train, w)-y_train, 2) / 471 / 12))

    if time % 100 == 0:
        # 每迭代100次就输入一次损失
        print(f"{time}次之后的损失为{loss}")

    gradient = 2 * np.dot(x_train.transpose(), np.dot(x_train, w)-y_train)
    adagram += gradient**2
    w -= learning_rate * gradient / np.sqrt(np.sum(adagram + eps))
    
    
np.save("weight_self.np", w)


# 载入验证集进行验证
w_verify = np.load("./weight_self.np.npy")

x_test_set = np.concatenate((np.ones([len(x_test_set), 1]), x_test_set), axis=1)

y_verify = np.dot(x_test_set, w_verify)

loss_verify = np.sqrt(np.sum(np.power(y_verify - y_test_set, 2) / len(y_verify)))
print("loss_verify=", loss_verify)

# 预测结果
data_test = pd.read_csv("/Users/jingrancao/Downloads/数据/hw1/test.csv", encoding="big5", header=None)
data_test.replace(to_replace="NR", value=0, inplace=True)
data_test_p = data_test.iloc[:, 2:]
data_test_np = data_test_p.to_numpy()
data_test_x = np.empty([240, 18*9])
for id in range(240):
    data_test_x[id, :] = data_test_np[18*id: 18*(id+1), :].reshape(1, -1)

# normalization:
data_test_std = np.std(data_test_x, axis=0)
data_test_mean = np.mean(data_test_x, axis=0)

for i in range(len(data_test_x)):
    for j in range(len(data_test_x[0])):
        if data_test_std[j] == 0:
            data_test_std[j] = 0.00001
        data_test_x[i, j] = (data_test_x[i, j] - data_test_mean[j]) / data_test_std[j]

data_text_x_f = np.concatenate((np.ones([len(data_test_x), 1]), data_test_x), axis=1)

w = np.load("weight_self.np.npy")
predict_y = np.dot(data_text_x_f, w)

with open("./final_result.csv", mode="w", newline="") as final_result:
    writer = csv.writer(final_result)
    header = ["id", "PM2.5_value"]
    writer.writerow(header)
    for i in range(240):
        row = ["id_" + str(i), predict_y[i][0]]  # 需要切片 否则会写入[]
        writer.writerow(row)






    











