import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

X = np.array([
    [0.5, 0.5],
    [1, 1],
    [1, -0.5],
    [-0.5, -0.5],
    [2, 2],
    [4, 0],
    [4.5, 1],
    [3.5, 5],
    [5, 1],
    [5, 2]
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

clf = svm.SVC(kernel='linear', C=1e5)
clf.fit(X, y)

support_vectors = clf.support_vectors_
w = clf.coef_[0]
b = clf.intercept_[0]

print("Support Vectors:\n", support_vectors)
print("Hyperplane weights (w):", w)
print("Intercept (b):", b)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = (-w[0] * xx - b) / w[1]

plt.plot(xx, yy)
plt.show()