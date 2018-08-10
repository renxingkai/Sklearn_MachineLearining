import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import time

#预测乳腺癌患病
cancer=load_breast_cancer()
#获取X和y
X=cancer.data
y=cancer.target

#划分测试集和训练集
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

#构建模型进行训练
model=LogisticRegression(penalty='l2')
model.fit(X_train,y_train)
#计算训练集的拟合分数
train_score=model.score(X_train,y_train)

test_score=model.score(X_test,y_test)

print('train score:',train_score,'test score:',test_score)

print('train score:{train_score:.6f},test score:{test_score:.6f}'.format(train_score=train_score,test_score=test_score))

#计算样本预测值
y_pred=model.predict(X_test)

#全部预测正确，但是test值不到100%原因是,sklearn不根据此来计算数据分数
print('matches:{0}/{1}'.format(np.equal(y_pred,y_test).shape[0],y_test.shape[0]))

#预测概率，找出预测概率低于90%的样本
y_pred_proba=model.predict_proba(X_test)

# print(y_pred_proba[0])

y_pred_proba_0=y_pred_proba[:,1]>0.1
#下面输出结果为：
'''
[False  True  True  True  True  True  True  True  True  True  True  True
  True  True  True False  True False False False False False  True  True
 False  True  True  True  True False  True False  True False  True False
  True False  True False False  True False  True False False  True  True
  True False False  True False  True  True  True  True  True  True False
 False False  True  True False  True False False False  True  True False
  True  True False  True  True  True  True  True False False False  True
 False  True  True  True False False  True  True  True False  True  True
 False  True  True  True  True  True  True  True False  True False  True
  True  True  True False False  True]
'''
# print(y_pred_proba_0)

result=y_pred_proba[y_pred_proba_0]
#下式输出结果:
'''
[[3.00726014e-02 9.69927399e-01]
 [2.07974449e-03 9.97920256e-01]
 [1.82556287e-01 8.17443713e-01]
 [7.73604563e-05 9.99922640e-01]
 [2.91254234e-03 9.97087458e-01]
 [7.19546821e-03 9.92804532e-01]
 [1.61297803e-03 9.98387022e-01]
 [3.41462388e-02 9.65853761e-01]
 [2.04245833e-04 9.99795754e-01]
 [3.90516252e-01 6.09483748e-01]
 [1.58327182e-01 8.41672818e-01]
 [3.95747573e-03 9.96042524e-01]
 [7.64658052e-01 2.35341948e-01]
 [1.96876886e-01 8.03123114e-01]
 [1.49140225e-02 9.85085978e-01]
 [1.64129929e-03 9.98358701e-01]
 [1.09552569e-02 9.89044743e-01]
'''
# print(result)

#两次值都大于0.1，最终小于90%
y_pred_proba_1=result[:,0]>0.1

greaterThan90=result[y_pred_proba_1]

print(greaterThan90)


#定义与多项式结合的优化LogisticRegression
def polynomial_model(degree,**kwargs):
    polynomial_features=PolynomialFeatures(degree=degree,include_bias=False)
    logistic_regression=LogisticRegression(**kwargs)
    pipeline=Pipeline([('polynomial_features',polynomial_features),('logistic_regression',logistic_regression)])
    return pipeline

#多项式联合模型,使用L1正则化
polynomialModel=polynomial_model(degree=2,penalty='l1')

startTime=time.clock()
#进行模型的训练
polynomialModel.fit(X_train,y_train)
#训练模型进行打分
train_score=polynomialModel.score(X_train,y_train)
#测试模型进行打分
test_score=polynomialModel.score(X_test,y_test)
#输出结果
print('elapse:{0:.6f},train score:{train_score:.6f},test score:{test_score:.6f}'.format((time.clock()-startTime),train_score=train_score,test_score=test_score))
#查询使用了多少特征、多少特征被丢弃
logistic_regression=polynomialModel.named_steps['logistic_regression']
#输出非0元素特征
print('model parameters shape:{0};count of non-zero element:{1}'.format(logistic_regression.coef_.shape,np.count_nonzero(logistic_regression.coef_)))






















