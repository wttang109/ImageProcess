# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:16:30 2018

@author: User
"""

import pandas as pd
from sklearn import cross_validation, ensemble, preprocessing, metrics
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import os
os.environ['PATH'] = os.environ['PATH'] + (';C:\\ProgramData\\Anaconda3\\Lib\\site-packages\\')
#import os
#os.environ["PATH"] += os.pathsep + 'G:/program_files/graphviz/bin'

# 載入資料
url = "max_data.xlsx"
max_abs = pd.read_excel(url)

# 建立訓練與測試資料
max_abs_X = pd.DataFrame([max_abs["f_150L"],
                          max_abs["d_150L"],
#                          max_abs["f_100L"],
                          max_abs["d_100L"],
#                          max_abs["f_75L"],
                          max_abs["d_75L"],
]).T #行列互換
max_abs_y = max_abs["type"]
train_X, test_X, train_y, test_y = cross_validation.train_test_split(max_abs_X, max_abs_y, test_size = 0.3)

# 建立 random forest 模型
forest = tree.DecisionTreeClassifier()
 #决策树的个数，越多越好，但是性能就会越差，至少100左右（具体数字忘记从哪里来的了）可以达到可接受的性能和误差率。 

forest = forest.fit(train_X, train_y)

######## 提取单棵决策树
#tree =forest.estimators_[5]

# 训练完成后，我们可以用 export_graphviz 将树导出为 Graphviz 格式，存到文件iris.dot中
'''
with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(forest, out_file=f)
'''

dot_data = tree.export_graphviz(forest, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('iris.pdf')





'''
# 输出一个.dot格式的文件
export_graphviz(tree,out_file='tree.dot',
                feature_names =max_abs.feature_names,
                class_names =max_abs.target_names,
                rounded = True, proportion =False,
                precision = 2, filled = True)
'''


'''

# 預測
test_y_predicted = forest.predict(test_X)

# 績效
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)

# 績效
fpr, tpr, thresholds = metrics.roc_curve(test_y, test_y_predicted)
auc = metrics.auc(fpr, tpr)
print(auc)
'''




############ 原文：https://blog.csdn.net/ydyang1126/article/details/78842952 
'''
os.environ["PATH"] += os.pathsep + 'G:/program_files/graphviz/bin'

# 仍然使用自带的iris数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 训练模型，限制树的最大深度4
clf = RandomForestClassifier(max_depth=4)
#拟合模型
clf.fit(X, y)

Estimators = classifier.estimators_
for index, model in enumerate(Estimators):
    filename = 'iris_' + str(index) + '.pdf'
    dot_data = tree.export_graphviz(model , out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # 使用ipython的终端jupyter notebook显示。
    Image(graph.create_png())
    graph.write_pdf(filename)

'''




