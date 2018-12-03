# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:16:30 2018

@author: sunny
"""

import pandas as pd
from sklearn import cross_validation, ensemble, metrics
from sklearn.tree import DecisionTreeClassifier

# loading data
url = "kurt_skew.csv"
max_abs = pd.read_csv(url)

# build train ans test data
max_abs_X = pd.DataFrame([max_abs["kt_c_150L"],max_abs["kt_d_150L"],max_abs["kt_e_150L"],max_abs["kt_f_150L"],
                          max_abs["kt_g_150L"],max_abs["kt_h_150L"],max_abs["kt_i_150L"],max_abs["kt_j_150L"],
                          max_abs["kt_c_100L"],max_abs["kt_d_100L"],max_abs["kt_e_100L"],max_abs["kt_f_100L"],
                          max_abs["kt_g_100L"],max_abs["kt_h_100L"],max_abs["kt_i_100L"],max_abs["kt_j_100L"],
                          max_abs["kt_c_75L"], max_abs["kt_d_75L"], max_abs["kt_e_75L"], max_abs["kt_f_75L"],
                          max_abs["kt_g_75L"], max_abs["kt_h_75L"], max_abs["kt_i_75L"], max_abs["kt_j_75L"],
                          max_abs["sk_c_150L"],max_abs["sk_d_150L"],max_abs["sk_e_150L"],max_abs["sk_f_150L"],
                          max_abs["sk_g_150L"],max_abs["sk_h_150L"],max_abs["sk_i_150L"],max_abs["sk_j_150L"],
                          max_abs["sk_c_100L"],max_abs["sk_d_100L"],max_abs["sk_e_100L"],max_abs["sk_f_100L"],
                          max_abs["sk_g_100L"],max_abs["sk_h_100L"],max_abs["sk_i_100L"],max_abs["sk_j_100L"],
                          max_abs["sk_c_75L"], max_abs["sk_d_75L"], max_abs["sk_e_75L"], max_abs["sk_f_75L"],
                          max_abs["sk_g_75L"], max_abs["sk_h_75L"], max_abs["sk_i_75L"], max_abs["sk_j_75L"]]).T

# swich row and colum
max_abs_y = max_abs["label_list"]

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
url_test = 'kurt_MFP_good_1_MFP_60T-brokengear_3.csv'
test2 = pd.read_csv(url_test)

test2_X = pd.DataFrame([test2["kt_c_150L"],test2["kt_d_150L"],test2["kt_e_150L"],test2["kt_f_150L"],
                        test2["kt_g_150L"],test2["kt_h_150L"],test2["kt_i_150L"],test2["kt_j_150L"],
                        test2["kt_c_100L"],test2["kt_d_100L"],test2["kt_e_100L"],test2["kt_f_100L"],
                        test2["kt_g_100L"],test2["kt_h_100L"],test2["kt_i_100L"],test2["kt_j_100L"],
                        test2["kt_c_75L"], test2["kt_d_75L"], test2["kt_e_75L"], test2["kt_f_75L"],
                        test2["kt_g_75L"], test2["kt_h_75L"], test2["kt_i_75L"], test2["kt_j_75L"],
                        test2["kt_c_150L"],test2["kt_d_150L"],test2["kt_e_150L"],test2["kt_f_150L"],
                        test2["kt_g_150L"],test2["kt_h_150L"],test2["kt_i_150L"],test2["kt_j_150L"],
                        test2["kt_c_100L"],test2["kt_d_100L"],test2["kt_e_100L"],test2["kt_f_100L"],
                        test2["kt_g_100L"],test2["kt_h_100L"],test2["kt_i_100L"],test2["kt_j_100L"],
                        test2["kt_c_75L"], test2["kt_d_75L"], test2["kt_e_75L"], test2["kt_f_75L"],
                        test2["kt_g_75L"], test2["kt_h_75L"], test2["kt_i_75L"], test2["kt_j_75L"]]).T

test_y = test2["label"]
test_y_predicted = forest.predict(test2_X)
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)
'''


'''
# accuracy
fpr, tpr, thresholds = metrics.roc_curve(test_y, test_y_predicted)
auc = metrics.auc(fpr, tpr)
print(auc)
'''





