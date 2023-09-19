from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

import dataReader as data
import train as train

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
   # print(xx1)
    print(xx2.shape)
   # print(xx2)
    
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

plot_decision_regions(data.X, data.y, train.ppn, resolution=0.02)
plt.xlabel('huajing length')
plt.ylabel('huaban length')
plt.legend(loc='upper left')
plt.show()