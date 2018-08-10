import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#sklearn提供了模型选择和评估的工具
from sklearn.model_selection import GridSearchCV



#训练集文件路径
trainFileName='C:/Users/Administrator/Desktop/all/train.csv'

#读取数据
def read_dataset(fileName):
    '''
    粗糙的数据处理
    :param fileName:
    :return:
    '''
    #指定第一列作为索引
    data=pd.read_csv(fileName,index_col=0)
    #丢弃无用的数据
    data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
    #处理性别数据
    data['Sex']=(data['Sex']=='male').astype('int')
    #处理登船港口数据
    labels=data['Embarked'].unique().tolist()
    data['Embarked']=data['Embarked'].apply(lambda n:labels.index(n))
    #处理缺失数据
    data=data.fillna(0)
    return data


train=read_dataset(trainFileName)
y=train['Survived'].values
X=train.drop(['Survived'],axis=1).values

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

#(891, 7) (891,)
print(X.shape,y.shape)


#创建模型并进行训练
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
#evalution
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)

print('train score:{0},test score:{1}'.format(train_score,test_score))


#对模型进行调参
#max_depth
def cv_score(d):
    #构建模型进行训练
    model=DecisionTreeClassifier(max_depth=d)
    model.fit(X_train,y_train)
    train_score=model.score(X_train,y_train)
    test_score=model.score(X_test,y_test)
    return (train_score,test_score)

#构造参数范围，在这个范围内分别计算模型评分，并找出评分最高的模型所对应的参数
depths=range(2,15)
scores=[cv_score(d) for d in depths]
#统计每个训练集和测试集分数
tr_scores=[s[0] for s in scores]
cv_score=[s[1] for s in scores]

#找出交叉验证数据集评分最高的索引
best_score_index=np.argmax(cv_score)
#找出交叉验证数据集评分最高的分数
best_score=cv_score[best_score_index]
#最好测试集对应的depths深度索引参数
best_param=depths[best_score_index]

print('best param:{0};best score:{1};best score index:{2}'.format(best_param,best_score,best_score_index))


#利用sklearn提供的默认数据模型选择器进行模型选择与评估
thresholds=np.linspace(0,0.5,50)
#设置参数矩阵
#min_impurity_split为信息增益的阈值
param_grid={'min_impurity_split':thresholds}

#构建模型
clf=GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5)
clf.fit(X,y)
print('best param:{0}\nbest score:{1}\n'.format(clf.best_params_,clf.best_score_))



#做出模型参数与评分关系图
def plot_curve(train_sizes,cv_results,xlabel):
    train_score_mean=cv_results['mean_train_score']
    train_score_std=cv_results['std_train_score']
    test_score_mean=cv_results['mean_test_score']
    test_score_std=cv_results['std_test_score']
    plt.figure(figsize=(6,4),dpi=144)
    plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(train_sizes,train_score_mean-train_score_std,train_score_mean+train_score_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes,test_score_mean-test_score_std,test_score_mean+test_score_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_score_mean,'.--',color='r',label='Training score')
    plt.plot(train_sizes,test_score_mean,'.-',color='g',label='Cross-validation score')
    plt.legend(loc='best')
    plt.show()

plot_curve(thresholds,clf.cv_results_,xlabel='gini thresholds')

#在多组参数之间选择最优的参数
entropy_thresholds=np.linspace(0,1,50)
gini_thresholds=np.linspace(0,0.5,50)

#设置参数矩阵
'''
criterion:选择信息熵还是gini
min_impurity_split:阈值从[0,1]之间划分了50等分
max_depth:最大深度
min_samples_split:创建分支集的最小大小
'''
param_grid=[{'criterion':['entropy'],
             'min_impurity_split':entropy_thresholds},
            {'criterion':['gini'],
             'min_impurity_split':gini_thresholds},
            {'max_depth':range(2,10)},
            {'min_samples_split':range(2,30,2)}]

clf=GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5)
clf.fit(X,y)
print('\nbest param:{0};\n\nbest score:{1}'.format(clf.best_params_,clf.best_score_))

















