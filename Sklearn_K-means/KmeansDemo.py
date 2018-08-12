import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans




#产生样本
X,y=make_blobs(n_samples=200,n_features=2,centers=4,
               cluster_std=1,center_box=(-10.0,10.0),
               shuffle=True,random_state=1)

#做出样本图像
plt.figure(figsize=(6,4),dpi=1440)
#人为设置坐标轴的刻度显示的值，此处为不显示任何值
plt.xticks(())
plt.yticks(())
#做出散点图
plt.scatter(X[:,0],X[:,1],s=20,marker='o')
# plt.show()

'''
使用Kmeans进行拟合
设置类别数量为3
'''
n_cluster=3
kmean=KMeans(n_clusters=n_cluster)
kmean.fit(X)
print("kmean:k={},cost={}".format(n_cluster,int(kmean.score(X))))


'''
对分类后的样本机器所属的聚类中心绘制
'''
labels=kmean.labels_
print(labels)
'''
[0 1 2 2 0 1 1 1 2 0 1 0 2 0 0 2 0 2 0 2 0 0 2 1 0 1 1 2 2 0 0 2 2 2 0 0 2
 0 2 0 0 0 1 1 2 2 1 2 0 1 0 2 0 1 2 2 0 0 0 1 1 2 0 0 0 0 0 0 2 0 0 1 0 2
 1 0 2 2 2 2 0 1 2 2 0 0 0 1 2 2 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 2 0 2 1 0
 2 1 1 0 1 1 0 1 0 0 2 0 0 0 1 0 0 0 0 1 0 2 1 1 2 2 2 1 1 0 0 0 1 2 2 0 0
 1 0 2 1 1 0 1 0 0 1 0 1 0 0 0 1 2 0 0 0 2 1 0 0 2 2 1 0 1 1 0 0 0 0 1 1 2
 0 0 1 0 0 0 0 0 2 1 2 2 0 0 2]
'''
centers=kmean.cluster_centers_
print(centers)
'''
[[-8.03529126 -3.42354791]
 [-1.54465562  4.4600113 ]
 [-7.15632049 -8.05234186]]
'''

#点的标记
marker=['o','^','*']
#点的颜色
colors=['r','y','b']

plt.figure(figsize=(6,4),dpi=144)
plt.xticks(())
plt.yticks(())

#画样本
for c in range(n_cluster):
    #判断聚类label
    cluster=X[labels==c]
    plt.scatter(cluster[:,0],cluster[:,1],marker=marker[c],s=20,c=colors[c])
    # plt.show()
#画出中心点
plt.scatter(centers[:,0],centers[:,1],marker='o',c='white',alpha=0.9,s=300)
# plt.show()

for i,c in enumerate(centers):
    plt.scatter(c[0],c[1],marker='$%d$'%i,s=50,c=colors[i])
    # plt.show()

'''
将kmean算法进行聚类拟合，封装成函数
'''
def fit_plot_kmean_model(n_cluster,X):
    plt.xticks(())
    plt.yticks(())

    #使用kmeans算法进行拟合
    kmean=KMeans(n_clusters=n_cluster)
    kmean.fit_predict(X)

    labels=kmean.labels_
    centers=kmean.cluster_centers_
    markers=['o','^','*','s']
    colors=['r','b','y','k']

    #计算成本
    score=kmean.score(X)
    plt.title('k={},score={}'.format(n_cluster,(int)(score)))

    #画样本
    for c in range(n_cluster):
        cluster=X[labels==c]
        plt.scatter(cluster[:,0],cluster[:,1],marker=markers[c],colors=colors[c],s=20)

    #画出中心点
    plt.scatter(centers[:,0],centers[:,1],marker=markers[c],c='white',alpha=0.9,s=300)

    for i,c in enumerate(centers):
        plt.scatter(c[0], c[1], marker='$%d$' % i, s=50, c=colors[i])

'''
利用以上函数，对[2,3,4]3种不同的k值情况进行聚类分析，并将聚类结果可视化
'''

n_cluster=[2,3,4]

plt.figure(figsize=(10,3),dpi=144)
for i,c in enumerate(n_cluster):
    plt.subplot(1,3,i+1)
    fit_plot_kmean_model(c,X)


