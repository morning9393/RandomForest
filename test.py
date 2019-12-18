from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import scipy.stats as st
import DecisionTree
import RandomForest
import time


def plot_sk_dt(data, k, params):
    train_label = data[data.columns[0]]
    train_data = data[data.columns[1:]]

    train_data = train_data.values.tolist()
    train_label = train_label.tolist()

    results = []
    for param in params:
        dtc = DecisionTreeClassifier(min_samples_leaf=param[0], max_depth=param[1], criterion=param[3])
        cv_scores = cross_val_score(dtc, train_data, train_label, cv=k)
        mean = np.mean(cv_scores)
        interval = st.norm.interval(0.95, loc=mean, scale=st.sem(cv_scores))
        result = [param, mean, interval]
        results.append(result)
    return results


def plot_sk_rf(data, k, params):
    train_label = data[data.columns[0]]
    train_data = data[data.columns[1:]]

    train_data = train_data.values.tolist()
    train_label = train_label.tolist()

    results = []
    for param in params:
        rfc = RandomForestClassifier(n_estimators=param[0], min_samples_split=param[1])
        cv_scores = cross_val_score(rfc, train_data, train_label, cv=k)
        mean = np.mean(cv_scores)
        interval = st.norm.interval(0.95, loc=mean, scale=st.sem(cv_scores))
        result = [param, mean, interval]
        results.append(result)
    return results


train = pd.read_csv('test/income_train_500.csv')
test = pd.read_csv('test/income_test_50.csv')

rf = RandomForest.RandomForest(n_trees=10, min_sample_split=2, n_features='sqrt', criterion='gini')
t1 = time.time()
rf.fit(train)
t2 = time.time()
print('Fitting time cost: %f' % (t2 - t1))
score = rf.score(test)
print('Score: %f' % score)

# dt = DecisionTree.DecisionTree(min_sample_split=20, max_depth=20, n_features='all', criterion='gini')
# t1 = time.time()
# dt.fit(train)
# t2 = time.time()
# print('Fitting time cost: %f' % (t2 - t1))
# score = dt.score(test)
# print('score: %f' % score)
