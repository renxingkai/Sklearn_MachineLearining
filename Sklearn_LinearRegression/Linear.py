import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from matplotlib.figure import SubplotParams

n_dots=200
#在[-2pi,2pi]生成200个点，并加入随机噪声
X=np.linspace(-2*np.pi,2*np.pi,n_dots)
y=np.sin(X)+np.random.rand(n_dots)-0.1
#将数据转为sklearn便于输入的形式
'''
key
'''
X=X.reshape(-1,1)
y=y.reshape(-1,1)

def polynomial_model(degree=1):
    '''
    创建一个多项式和线性回归拟合模型pipeline
    :param degree:
    :return:
    '''
    polynomial_features=PolynomialFeatures(degree=degree,include_bias=False)
    #对线性模型进行归一化处理
    linear_Regression=LinearRegression(normalize=True)
    pipeline=Pipeline([('polynomial_features',polynomial_features),('linear_Regression',linear_Regression)])
    return pipeline

#分别使用2、3、5、10阶多项式来拟合
#定义多项式的阶数
degrees=[2,3,5,10]
results=[]

for d in degrees:
    model=polynomial_model(degree=d)
    #进行模型的训练
    model.fit(X,y)
    #训练数据集的得分
    train_score=model.score(X,y)
    #计算y和预测值的均方差误差
    mse=mean_squared_error(y,model.predict(X))
    results.append({'model':model,'degree':d,'score':train_score,'mse':mse})

for r in results:
    print('degree:{};train score:{};mean square error:{}'.format(r['degree'],r['score'],r['mse']))


#做出拟合图像
plt.figure(figsize=(12,6),dpi=200,subplotpars=SubplotParams(hspace=0.3))

for i,r in enumerate(results):
    #(2,2)子图，依次绘制
    fig=plt.subplot(2,2,i+1)
    #x轴范围
    plt.xlim(-8,8)
    #图标题
    plt.title('LinearRegression degree={}'.format(r['degree']))
    #真实（带噪声）散点图绘制，c为颜色,s为
    plt.scatter(X,y,s=5,c='b',alpha=0.5)
    plt.plot(X,r['model'].predict(X),'r-')
    plt.show()

