import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === 1. 读取数据 ===
path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# === 2. 可视化原始数据 ===
data.plot(kind='scatter', x='Population', y='Profit', figsize=(10,8))
plt.show()

# === 3. 构造 X 和 y ===
data.insert(0, 'Ones', 1)  # 在第0列插入一列全1，用来表示截距项
X = data[['Ones', 'Population']].values  # 两列：常数项 + 特征
y = data[['Profit']].values

X = np.matrix(X)
y = np.matrix(y)
theta = np.zeros((2, 1))  # θ0, θ1，均初始化为0

# === 4. 定义代价函数 ===
def computeCost(X, y, theta):
    inner = np.power((X @ theta - y), 2)
    return np.sum(inner) / (2 * len(X))

# === 5. 定义梯度下降函数 ===
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.zeros(theta.shape)
    parameters = theta.shape[0]
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = X @ theta - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[j, 0] = theta[j, 0] - ((alpha / len(X)) * np.sum(term))
        
        theta = temp.copy()
        cost[i] = computeCost(X, y, theta)
    
    return theta, cost

# === 6. 执行梯度下降 ===
alpha = 0.01
iters = 1000
g, cost = gradientDescent(X, y, theta, alpha, iters)

print("训练后参数：")
print(g)

# === 7. 绘制拟合曲线 ===
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + g[1, 0] * x  # 正确的预测函数

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
