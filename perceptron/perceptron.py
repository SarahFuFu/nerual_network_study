import numpy as np

class Perceptron(object):
    """
    eta: 学习率
    n_iter: 权重向量的训练次数
    w_: 神经分叉权重向量
    errors_: 用于记录神经元判断出错次数
    """
    def __init__(self,eta=0.01, n_iter=10):
        self.eta = eta;
        self.n_iter = n_iter
        pass
    def fit(self, X, y):  #训练算法
        """
        输入神经训练数据，培训神经元，X：输入电信号向量, y: 电信号对应的样本分类
        X:shape[n_samples, n_features] n_samples: 样本向量数， n_features: 神经分叉数
        X: [[1,2,3],[4,5,6]]    n_samples:2, n_features: 3
        y:[1, -1]
        """ 
        
        """
        初始化权重向量为0
        加1是因为前面算法提到的w0， 也就是步调函数阈值
        
        """
        self.w_ = np.zeros(1 + X.shape[1]);
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            """
            X: [[1,2,3],[4,5,6]]    y:[1, -1]
            zip(X,y) = [[[1,2,3, 1],[4,5,6, -1]]]
            """
            for xi, target in zip(X,y):
                """
                update = η*(y-y')
                """
                update = self.eta * (target - self.predict(xi))
                
                """
                xi是一个向量, update是一个标量
                update * xi 等价：
                [▽w(1)=X[1]*update,▽w(2)=X[2]*update, ▽w(3)=X[3]*update]
                """
                self.w_[1:] += update * xi
                self.w_[0] += update
                
                errors += int(update != 0.0)
            self.errors_.append(errors)
                
    def net_input(self, X):
        """
        z=W0*1 + W1*X1 + ... + Wn*Xn
        """
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)