import matplotlib.pyplot as plt

import perceptron as perceptron
import dataReader as data

ppn = perceptron.Perceptron(eta=0.1, n_iter=10)
ppn.fit(data.X, data.y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('error type times')
plt.show()