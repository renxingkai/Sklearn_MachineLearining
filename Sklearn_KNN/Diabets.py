import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest


#加载数据
#Outcome 0未患病
#Outcome 1患病
data=pd.read_csv('diabetes.csv')
# print(data.shape)
# print(data.head())
# print(data.groupby('Outcome').size())

#分离X和y标签
# iloc按列取
X=data.iloc[:,0:8]

y=data.iloc[:,8]

print(X.shape)
print(y.shape)

#划分训练集和测试集
#测试集为0.2
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

models=[]
#添加KNN
models.append(('KNN',KNeighborsClassifier(n_neighbors=2)))
models.append(('KNN with weights',KNeighborsClassifier(n_neighbors=2,
                                                       weights='distance')))
models.append(('Radius Neighbors',RadiusNeighborsClassifier(n_neighbors=2,radius=500.0)))

#分别训练3个模型，并计算评分
# results=[]
# for name,model in models:
#     #将数据切分为10份，其中一份当做验证数据集，剩余九份当做训练数据集
#     kfold=KFold(n_splits=10)
#     cv_result=cross_val_score(model,X,y,cv=kfold)
#     results.append((name, cv_result))
#     # model.fit(X_train,y_train)
#     # results.append((name,model.score(X_test,y_test)))
#
# for i in range(len(results)):
#     print(results[i][0],results[i][1].mean())


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

knn=KNeighborsClassifier(n_neighbors=2)
cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)

knn.fit(X_train,y_train)
print('训练集准确率:',knn.score(X_train,y_train))

print('测试集准确率:',knn.score(X_test,y_test))

#选取相关性最大的两个特征放在X_new
selector=SelectKBest(k=2)
X_new=selector.fit_transform(X,y)
print(X_new[:5])


#仅选择这两个相关性最大的特征对三个模型进行训练
result=[]
for name,model in models:
    #10折交叉验证
    kflod=KFold(n_splits=10)
    cv_result=cross_val_score(model,X_new,y,cv=cv)
    result.append((name,cv_result))

#对结果进行输出
for i in range(len(result)):
    print(result[i][0],result[i][1].mean())

print(X_new[y==0][:5])
print(X_new[y==1][:5])

#对两个重要的特征图像进行绘制
plt.figure(figsize=(10,6),dpi=200)
plt.ylabel('BMI')
plt.xlabel('Glucose')
#画出Y==0的阴性样本，圆圈表示
plt.scatter(X_new[y==0][:,0],X_new[y==0][:,1],c='r',s=20,marker='o')
plt.scatter(X_new[y==1][:,0],X_new[y==1][:,1],c='g',s=20,marker='^')
plt.show()

