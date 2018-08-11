import numpy as np
import pandas as pd

#导入PCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

#需降维的数据
A=np.array([[3,2000],
            [2,3000],
            [4,5000],
            [5,8000],
            [1,2000]],dtype='float')

#进行数据预处理
def std_PCA(**argv):
    '''
    数据预处理
    :param argv:
    :return:
    '''
    scaler=MinMaxScaler()
    pca=PCA(**argv)
    pipeline=Pipeline([('scaler',scaler),('pca',pca)])
    return pipeline

#此处pipeline将数据预处理和PCA算法组成一个串行流水线
pca=std_PCA(n_components=1)
#进行数据的降维
R2=pca.fit_transform(A)

print(R2)
'''
[[-0.2452941 ]
 [-0.29192442]
 [ 0.29192442]
 [ 0.82914294]
 [-0.58384884]]
 和之前符号相反因为从不同的方向进行降维
'''

#进行数据恢复
#由于pipeline先PCA还原，再执行预处理
source=pca.inverse_transform(R2)
print(source)













