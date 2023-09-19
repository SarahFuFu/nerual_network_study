class AdalineGD(object):
    """
    eta: float 学习效率，处于0和1
    n_iter: 权重向量的训练次数
    w_: 神经分叉权重向量
    cost_: 一维向量，用于记录神经元判断代价
    """
    def __init__(self,eta=0.01, n_iter=50):
        self.eta = eta;
        self.n_iter = n_iter
        
    def fit(self, X, y):  #训练算法
        """
        X: 二维数组 [n_samples, n_features]  n_samples表示X中含有训练数据条目数，n_features含有几个数据的一维向量，用于表示一条训练条目
        y: 一维向量，用于存储每一条训练条目对应的正确分类
        """
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []
        self.errors_ = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors) #从下标为1开始的每一个w都加上 ▽w
            self.w_[0] += self.eta * errors.sum() # X0=1
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
            
    def net_input(self, X):
        """
        z=W0*1 + W1*X1 + ... + Wn*Xn
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return self.net_input(X)
    
    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1)

file="data.csv"
import pandas as pd
df = pd.read_csv(file, header = None)

import matplotlib.pyplot as plt
import numpy as np
y = df.loc[0:99, 4].values # 取第4列数据的0-100行, 注意loc函数的端点是闭区间
y = np.where(y=='Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values # 取第0列和第2列数据的0-100行

plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:150, 0], X[50:150, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('huaban length')
plt.ylabel('huajing length')
plt.legend(loc='upper left')
plt.show()

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()
    
    print(x1_min, x1_max)
    print(x2_min, x2_max)
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    print(xx1.shape)
    print(xx2.shape)
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print(xx1.ravel())
    print(xx2.ravel())
    print(Z)
    
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap) #画分隔线
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter( x=X[y==cl, 0], y=X[y==cl, 1],alpha=0.8, color=cmap(idx),
                    marker=markers[idx], label=cl)

ada = AdalineGD(eta=0.0001, n_iter=50)
ada.fit(X,y)

plot_decision_regions(X, y, classifier=ada)
plt.title('Adaline-Gradient descent')
plt.xlabel('huajing length')
plt.ylabel('huaban length')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('sum-squard-error')
plt.show()

