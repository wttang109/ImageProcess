# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:16:30 2018

@author: User
"""

import pandas as pd
from sklearn import cross_validation, ensemble, metrics
from sklearn.tree import DecisionTreeClassifier

# loading data
url = "max_data.xlsx"
max_abs = pd.read_excel(url)

# build train ans test data
max_abs_X = pd.DataFrame([max_abs["f_150L"],
                          max_abs["d_150L"],
                          max_abs["f_100L"],
                          max_abs["d_100L"],
                          max_abs["f_75L"],
                          max_abs["d_75L"],]).T # swich row and colum

max_abs_y = max_abs["type"]

train_X, test_X, train_y, test_y = cross_validation.train_test_split(max_abs_X, max_abs_y, test_size = 0.3)

# build random forest model
forest = ensemble.RandomForestClassifier(n_estimators = 100) # number of decision tree

forest = forest.fit(train_X, train_y)

# predicted
test_y_predicted = forest.predict(test_X)

# accuracy
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)
'''
url_test = 'max_MFP_good_1_MFP_60T-brokengear_2.csv'
test2 = pd.read_csv(url_test)
test2_X = pd.DataFrame([test2["f_150L"],
                        test2["d_150L"],
                        test2["f_100L"],
                        test2["d_100L"],
                        test2["f_75L"],
                        test2["d_75L"],]).T # swich row and colum

test2_predicted = forest.predict(test2_X)
print(test2_predicted)
'''

'''
# accuracy
fpr, tpr, thresholds = metrics.roc_curve(test_y, test_y_predicted)
auc = metrics.auc(fpr, tpr)
print(auc)
'''





