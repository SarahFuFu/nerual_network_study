import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file="data.csv"
df = pd.read_csv(file, header = None)
df.head(10)

y = df.loc[0:99, 4].values # 取第4列数据的0-100行, 注意loc函数的端点是闭区间
y = np.where(y=='Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values # 取第0列和第2列数据的0-100行

plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:150, 0], X[50:150, 1], color='blue', marker='x', label='versicolor')
# plt.scatter(X[100:150, 0], X[100:150, 1], color='green', marker='o',label='setosa')
plt.xlabel('huaban length')
plt.ylabel('huajing length')
plt.legend(loc='upper left')
plt.show()

