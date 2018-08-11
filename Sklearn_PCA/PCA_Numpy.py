import numpy as np

#需降维的数据
A=np.array([[3,2000],
            [2,3000],
            [4,5000],
            [5,8000],
            [1,2000]],dtype='float')

#数据归一化
#求出平均值
mean=np.mean(A,axis=0)
norm=A-mean
#数据缩放
scope=np.max(norm,axis=0)-np.min(norm,axis=0)
norm=norm/scope
print(norm)
'''
[[ 0.         -0.33333333]
 [-0.25       -0.16666667]
 [ 0.25        0.16666667]
 [ 0.5         0.66666667]
 [-0.5        -0.33333333]]
'''

#对协方差矩阵进行奇异值分节，求解特征向量
U,S,V=np.linalg.svd(np.dot(norm.T,norm))
print(U)
'''
[[-0.67710949 -0.73588229]
 [-0.73588229  0.67710949]]
 '''

#需要将二维数组降为一维，因此只取特征矩阵的第一列来构造出Ureduce
U_reduce=U[:,0].reshape(2,1)
print(U_reduce)
'''
[[-0.67710949]
 [-0.73588229]]
'''

#对数据进行降维
R=np.dot(norm,U_reduce)
print(R)
'''
[[ 0.2452941 ]
 [ 0.29192442]
 [-0.29192442]
 [-0.82914294]
 [ 0.58384884]]
'''

#还原原数据
Z=np.dot(R,U_reduce.T)
print(Z)
'''
[[-0.16609096 -0.18050758]
 [-0.19766479 -0.21482201]
 [ 0.19766479  0.21482201]
 [ 0.56142055  0.6101516 ]
 [-0.39532959 -0.42964402]]
'''

#进行数据预处理的逆运算
source=np.multiply(Z,scope)+mean
print(source)
'''
[[2.33563616e+00 2.91695452e+03]
 [2.20934082e+00 2.71106794e+03]
 [3.79065918e+00 5.28893206e+03]
 [5.24568220e+00 7.66090960e+03]
 [1.41868164e+00 1.42213588e+03]]
'''
