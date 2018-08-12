from __future__ import print_function
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from sklearn.datasets import load_files
from sklearn.cluster import KMeans


#导入文件
print('loading ducuments...')
t=time()
docs=load_files('clustering/')
print('summary:{0} documents in {1} categories.'.format(len(docs.data),len(docs.target_names)))
#导入文件完成
print('done in {0}'.format(time()-t))

'''
loading ducuments...
summary:7898 documents in 4 categories.
done in 0.9050517082214355
说明有7898文件，被标记在4个类别中
'''

'''
将这些文件转为TF-IDF向量
'''
max_features=20000
print('vectorizing documents...')
t=time()
vectorizer=TfidfVectorizer(max_df=0.4,
                           min_df=2,
                           max_features=max_features,
                           encoding='latin-1')
#转为TF-IDF向量
X=vectorizer.fit_transform((d for d in docs.data))
print('n_samples:%d,n_features:%d'%X.shape)
print('number of non-zero features in sample [{0}]:{1}'.format(
    docs.filenames[0],X[0].getnnz()
))
print('done in {0} seconds'.format(time()-t))
'''
n_samples:7898,n_features:20000
number of non-zero features in sample [clustering/sci.electronics\._12249-54259]:0
done in 1.9631125926971436 seconds
'''

'''
接下来使用Kmeans对文档进行聚类分析
'''
print('clustering documents...')
t=time()
n_clusters=4
kmean=KMeans(n_clusters=n_clusters,max_iter=100,tol=0.01,verbose=1,n_init=3)
kmean.fit(X)
print('kmean :k={},cost={}'.format(n_clusters,(int)(kmean.inertia_)))
print('done in {0} seconds'.format(time()-t))
'''
clustering documents...
Initialization complete
Iteration  0, inertia 3940.255
Iteration  1, inertia 3842.042
Iteration  2, inertia 3841.675
Iteration  3, inertia 3841.374
Iteration  4, inertia 3840.820
Iteration  5, inertia 3838.311
Iteration  6, inertia 3825.989
Iteration  7, inertia 3805.212
Iteration  8, inertia 3781.715
Iteration  9, inertia 3764.113
Iteration 10, inertia 3753.249
Iteration 11, inertia 3748.122
Iteration 12, inertia 3746.209
Iteration 13, inertia 3745.627
Iteration 14, inertia 3745.451
Iteration 15, inertia 3745.394
Iteration 16, inertia 3745.373
Iteration 17, inertia 3745.364
Iteration 18, inertia 3745.360
Iteration 19, inertia 3745.359
Converged at iteration 19: center shift 1.392080e-07 within tolerance 2.438758e-07
Initialization complete
Iteration  0, inertia 3944.424
Iteration  1, inertia 3846.122
Converged at iteration 1: center shift 0.000000e+00 within tolerance 2.438758e-07
Initialization complete
Iteration  0, inertia 3942.787
Iteration  1, inertia 3845.068
Converged at iteration 1: center shift 0.000000e+00 within tolerance 2.438758e-07
kmean :k=4,cost=3745
done in 14.565833330154419 seconds
'''

'''
查看每种类别文档中，其权限最高的10个单词
'''

print('Top terms per cluster:')
order_centroids=kmean.cluster_centers_.argsort()[:,::-1]
#获取词典单词
terms=vectorizer.get_feature_names()
for i in range(n_clusters):
    print('Cluster %d:'%i,end='')
    for ind in order_centroids[i,:10]:
        print(' %s'%terms[ind],end='')
    print()
'''
Top terms per cluster:
Cluster 0: it that for you edu be this on are have
Cluster 1: action eff we act as on optomistic bboard underestimate acg
Cluster 2: puff sni rf tools pd ftp co uk caltech csx
Cluster 3: uga mcovingt covington ai georgia michael 706 542 0358 30602
'''

