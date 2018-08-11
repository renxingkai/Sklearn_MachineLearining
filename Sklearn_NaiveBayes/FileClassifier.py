import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print('loading train dataset....')
t=time()

'''
news_train.data数组包含了所有文档的文本信息
news_train.target数组包含了所有文档所属的类别
news_train.target_names包含了文档所属类别的名称
'''
#load_files会将目录下所有文档都读入内存，并根据所在的子目录名称打上标签
news_train=load_files('datasets/379/train')
print('summary:{0} documents in {1} categories.'.format(len(news_train.data),len(news_train.target_names)))

print('done in {0} seconds'.format(time()-t))
print(news_train.target_names[news_train.target[0]])

#构建TFIDF向量
print('vectorizing train dataset...')
t=time()
vectorizer=TfidfVectorizer(encoding='latin-1')
X_train=vectorizer.fit_transform((d for d in news_train.data))
print('n_samples :%d,n_features:%d'%X_train.shape)
print('number of non-zero features in sample [{0}]:{1}'.format(news_train.filenames[0],X_train[0].getnnz()))

#X_train是13180*130274维度
print('done in {0} seconds'.format(time()-t))

#进行模型的训练
print('train models...'.format(time()-t))
t=time()

#news_train.target数组包含了所有文档所属的类别
y_train=news_train.target
#构建模型,alpha表示平滑参数
clf=MultinomialNB(alpha=0.0001)
#训练模型
clf.fit(X_train,y_train)
#训练集分数
train_score=clf.score(X_train,y_train)

print('train score:{0}'.format(train_score))
print('done in {0} seconds'.format(time()-t))

#加载测试集
print('loading test dataset...')
t=time()
news_test=load_files('datasets/379/test')
#输出测试集大小
print('summary :{0} documents in {1} categories.'.format(len(news_test.data),len(news_test.target_names)))

#将文档向量化
print('vectoring test dataset...')
t=time()
X_test=vectorizer.transform((d for d in news_test.data))
y_test=news_test.target

#输出测试文档大小
print('n_samples:%d,n_features:%d'%X_test.shape)
print('number of non-zero features in sample [{0}]:{1}'.format(news_test.filenames[0],X_test[0].getnnz()))
print('dine in %fs'%(time()-t))

#对第一篇文档进行预测
pred=clf.predict(X_test[0])

print('predict:{0} is in category {1}'.format(news_test.filenames[0],news_test.target_names[pred[0]]))
print('actually :{0} is in category {1}'.format(news_test.filenames[0],news_test.target_names[news_test.target[0]]))

#对整体测试集进行预测
print('predicting test dataset...')
t0=time()
pred=clf.predict(X_test)
print('done in %fs'%(time()-t0))
#使用classification_report()函数来查看以下针对每个类别的预测准确性
print('classification report on test set for classifier:')
print(clf)
print(classification_report(y_test,pred,target_names=news_test.target_names))

#通过confusion_matrix()函数生成混淆矩阵，观察每种类别被错误分类的情况，例如这些被错误分类的文档是被错误分类到哪些类别里的
cm=confusion_matrix(y_test,pred)
print('confusion matrix:')
print(cm)



















