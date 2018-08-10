import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve


'''
做出学习曲线
'''
def plot_learning_curve(plt, estimator, title, X, y, ylim=None, cv=None,
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
    plt.plot(train_sizes, train_scores_mean, 'o--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt

def plot_param_curve(plt, train_sizes, cv_results, xlabel):
    train_scores_mean = cv_results['mean_train_score']
    train_scores_std = cv_results['std_train_score']
    test_scores_mean = cv_results['mean_test_score']
    test_scores_std = cv_results['std_test_score']
    plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, '.--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, '.-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt

#载入数据
cancer=load_breast_cancer()
X=cancer.data
y=cancer.target
#划分测试、数据集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print('data shape:{0};no.positive:{1};no.negtive:{2}'.format(X.shape,y[y==0].shape,y[y==1].shape))

#数据量不大,可能用高斯核函数会出现过拟合
clf=SVC(C=1.0,kernel='rbf',gamma=0.1)
clf.fit(X_train,y_train)
#训练数据评分
print('train score:{0}'.format(clf.score(X_train,y_train)))
#测试数据评分
print('test score:{0}'.format(clf.score(X_test,y_test)))

'''
train score:1.0
test score:0.5877192982456141
明显出现过拟合
'''

#对参数gamma进行改进,多次迭代
gammas=np.linspace(0,0.0003,30)
param_grid={'gamma':gammas}
#高斯核函数
clf=GridSearchCV(SVC(kernel='rbf'),param_grid,cv=5)
#直接对所有数据进行训练
clf.fit(X,y)
print('best param:{0}\nbest score:{1}'.format(clf.best_params_,clf.best_score_))
'''
输出结果
best param:{'gamma': 0.00011379310344827585}
best score:0.9367311072056239
'''

#做出学习曲线观察模型
cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
title='Learning Curves for Gaussian Kernel'

startTime=time.clock()
# plt.figure(figsize=(10,4),dpi=144)
# plot_learning_curve(plt,SVC(C=1.0,kernel='rbf',gamma=0.01),
#                     title=title,X=X,y=y,ylim=(0.5,1.0),cv=cv)


'''
使用高斯核会出现过拟合，接下来使用多项式核函数来拟合模型
暂时使用2阶
'''

clf=SVC(C=1.0,kernel='poly',degree=2)
clf.fit(X_train,y_train)
#训练数据评分
print('poly train score:{0}'.format(clf.score(X_train,y_train)))
#测试数据评分
print('ploy test score:{0}'.format(clf.score(X_test,y_test)))












