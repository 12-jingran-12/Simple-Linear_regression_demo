import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.seterr(invalid='ignore')
"""
简单的一元线性规划 adagram
"""

# adagram 初始选择有随机性 数据量不够大导致结果波动较大
# 创建的数据集 噪音较大
# 最好记录 mse 四分位数 = 12 最小值 7.


#  创建数据

# x = np.random.rand(200)
# data = pd.DataFrame({"x": x,
#                      "y": x * 5 + 3 * np.random.randn(200)})
#
# data.to_csv("./train_test.csv")
data_init = pd.read_csv("./train_test.csv", header=0, index_col=0)
data = data_init.copy()

train = data.iloc[:150, :]
test = data.iloc[150:, :]

# y = wx + b


def loss_function(w, b, data_f):
    # 损失函数
    total_loss = 0
    for i in range(len(data_f)):
        x = data_f["x"]
        y = data_f["y"]
        loss = (y - (w*x + b)) ** 2
        total_loss += loss
    return total_loss / len(data_f)


def random_par(data_f):
    # 随机初始梯度
    random_list = np.random.randint(1, 150, 2)
    random_s1 = data_f.iloc[random_list[0], :]
    random_s2 = data_f.iloc[random_list[1], :]
    random_df = pd.concat([random_s1, random_s2], axis=1)
    w = (random_df.iloc[1, 0] - random_df.iloc[1, 1]) / (random_df.iloc[0, 0] - random_df.iloc[0, 1]+0.00000001)
    b = random_df.iloc[1, 0] - random_df.iloc[0, 0] * w
    return w, b, random_list[0]


def gradient_descent(w_current, b_current, begin, data_f, learningRate):
    w_gradient_list = []
    b_gradient_list = []
    N = float(len(data_f))
    while begin < len(data_f):
        x = data_f.iloc[begin, 0]
        y = data_f.iloc[begin, 1]
        b_gradient = -np.abs((2/N) * (y - ((w_current * x) + b_current)))
        w_gradient = -np.abs((2/N) * x * (y - ((w_current * x) + b_current)))
        w_gradient_list.append(w_gradient ** 2)
        b_gradient_list.append(b_gradient ** 2)
        while w_current == 0 and b_current == 0:
            w_current += w_gradient * learningRate / np.sqrt(sum(b_gradient_list))
            b_current += b_gradient * learningRate / np.sqrt(sum(w_gradient_list))
        begin += 1
    return w_current, b_current


def predict(test_f):
    result = test_f.copy()
    par = random_par(train)
    result_par = gradient_descent(par[0], par[1], par[2], train, 0.001)
    result["predict"] = result["x"]*result_par[0] + result_par[1]
    result["loss"] = (result["predict"] - result["y"]) ** 2 / len(result)
    return result


def run():
    result_f = predict(test)
    mse = result_f["loss"].sum()
    return mse


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    r_list = []
    for i in range(10000):
        r = run()
        r_list.append(r)
    r_np = np.array(r_list)
    r_np = r_np[r_np != 0]
    print(r_np.min())

