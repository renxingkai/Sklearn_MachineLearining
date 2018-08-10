import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
'''
scikit中的make_blobs方法常被用来生成聚类算法的测试数据，
直观地说，make_blobs会根据用户指定的特征数量、中心点数量、范围等来生成几类数据，
这些数据可用于测试聚类算法的效果。
'''
from sklearn.datasets import make_blobs

#画出样本点
def plot_hyperplane(clf, X, y,
                    h=0.02,
                    draw_sv=True,
                    title='hyperplan'):
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)

    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y)
    for label in labels:
        plt.scatter(X[y == label][:, 0],
                    X[y == label][:, 1],
                    c=colors[label],
                    marker=markers[label])
        plt.show()
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')
        plt.show()

#生成聚类数据
#生成两个特征，包含两种类别的数据集
X,y=make_blobs(n_samples=100,centers=2,random_state=0,cluster_std=0.3)

clf=svm.SVC(C=1.0,kernel='linear')
clf.fit(X,y)
plt.figure(figsize=(12,4),dpi=144)
plot_hyperplane(clf,X,y,h=0.01,title='Maximum Margin Hyperplan')


#生成两个特征，包含三种类别的数据集
X,y=make_blobs(n_samples=100,centers=3,random_state=0,cluster_std=0.8)
#分别使用线性核函数
clf_linear=svm.SVC(C=1.0,kernel='linear')
#三阶多项式核函数
clf_poly=svm.SVC(C=1.0,kernel='poly',degree=3)
#高斯核函数,gamma=0.5
clf_rbf=svm.SVC(C=1.0,kernel='rbf',gamma=0.5)
#高斯核函数,gamma=0.1
clf_rbf2=svm.SVC(C=1.0,kernel='rbf',gamma=0.1)

plt.figure(figsize=(10,10),dpi=144)
#训练模型集合
clfs=[clf_linear,clf_poly,clf_rbf,clf_rbf2]
titles=['Linear Kernel','Polynomial Kernel with Degree =3','Gaussian Kernel with $\gamma=0.5$','Gaussian Kernel with $\gamma=0.1$']

'''
zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
'''
#此处为[(clf,index)],eg:[(clf_linear,1)....]
for clf,i in zip(clfs,range(len(clfs))):
    print(zip(clfs,range(len(clfs))))
    clf.fit(X,y)
    plt.subplot(2,2,i+1)
    plot_hyperplane(clf,X,y,title=titles[i])

























