from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

#加载数据
boston=load_boston()
X=boston.data
y=boston.target


#切分训练集、测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

#数据变换程度较大，需要进行归一化
model=LinearRegression(normalize=True)
#记录开始时间
start=time.clock()
#进行模型的训练
model.fit(X_train,y_train)

#训练集分数
train_score=model.score(X_train,y_train)
#测试集分数
cv_score=model.score(X_test,y_test)

#输出训练时间，训练分数、测试分数等信息
print('elaspe:{0:.6f};train_score:{1:0.6f};cv_score:{2:.6f}'.format(time.clock()-start,train_score,cv_score))

#增加多项式特征
def polynomial_mode(degree=2):
    '''
    将多项式和线性模型进行结合
    :param degree:
    :return:
    '''
    polynomial_features=PolynomialFeatures(degree=degree,include_bias=False)
    linear_regression=LinearRegression(normalize=True)
    pipeline=Pipeline([('polynomial_features',polynomial_features),('linear_regression',linear_regression)])
    return pipeline

#使用2阶进行拟合
model=polynomial_mode(degree=2)
#记录开始时间
start=time.clock()
#进行模型训练
model.fit(X_train,y_train)
#训练分数
train_score=model.score(X_train,y_train)
#测试分数
test_score=model.score(X_test,y_test)
#输出训练时间，训练分数、测试分数等信息
print('elaspe:{0:.6f};train_score:{1:0.6f};cv_score:{2:.6f}'.format(time.clock()-start,train_score,test_score))


