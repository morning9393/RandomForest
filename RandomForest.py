import DecisionTree
import pandas as pd
import numpy as np
import scipy.stats as st
import time


class RandomForest:
    forest = []
    weights = []
    n_trees = 10
    min_sample_split = 2
    n_features = 'sqrt'
    criterion = 'gini'
    train_data = None
    label_col = None

    def __init__(self, n_trees=10, min_sample_split=2, n_features='sqrt', criterion='gini'):
        self.forest = []
        self.weights = []
        self.train_data = None
        self.label_col = None
        self.n_trees = n_trees
        self.min_sample_split = min_sample_split
        self.n_features = n_features
        self.criterion = criterion

    def fit(self, train_data):
        self.train_data = train_data
        self.label_col = train_data.columns[0]
        labels = list(set(train_data[self.label_col].tolist()))
        for n in range(0, self.n_trees):
            print('%d tree begin fit' % (n + 1))
            tt1 = time.time()
            train_labels = []
            for label in labels:
                train_label = train_data[train_data[self.label_col] == label]
                train_n_label = train_label.sample(frac=1, replace=True, random_state=20)
                train_labels.append(train_n_label)
            train_n = pd.concat(train_labels)
            train_n.reset_index(drop=True, inplace=True)
            tree = DecisionTree.DecisionTree(min_sample_split=self.min_sample_split,
                                             n_features=self.n_features,
                                             criterion=self.criterion)
            tree.fit(train_n)
            weight = tree.score(train_data)
            self.forest.append(tree)
            self.weights.append(weight)
            tt2 = time.time()
            print('%d tree time cost: %f' % (n + 1, tt2 - tt1))

    def predict_one(self, sample):
        if self.forest is None:
            print("Classifier should be fit first!")
            return None
        votes = []
        for tree in self.forest:
            votes.append(tree.predict_one(sample))
        weighted_votes = {}
        for i in range(0, len(votes)):
            if weighted_votes.get(votes[i]) is None:
                weighted_votes[votes[i]] = self.weights[i]
            else:
                weighted_votes[votes[i]] += self.weights[i]
        biggest_weight = 0
        predict = None
        for key in weighted_votes.keys():
            if weighted_votes[key] > biggest_weight:
                biggest_weight = weighted_votes[key]
                predict = key
        return predict

    def predict(self, samples):
        total = samples.shape[0]
        predicts = []
        for i in range(0, total):
            sample = samples.iloc[i]
            predicts.append(self.predict_one(sample))
        return predicts

    def score(self, test_data):
        total = test_data.shape[0]
        hit_counts = 0
        for i in range(0, total):
            sample = test_data.iloc[i]
            predict = self.predict_one(sample)
            label = sample[self.label_col]
            if predict == label:
                hit_counts += 1
        return hit_counts / total

    def confusion_matrix(self, test_data):
        total = test_data.shape[0]
        p_total = test_data[test_data[self.label_col] == 1].shape[0]
        n_total = test_data[test_data[self.label_col] == 0].shape[0]
        t_p = 0
        t_n = 0
        f_p = 0
        f_n = 0
        for i in range(0, total):
            sample = test_data.iloc[i]
            predict = self.predict_one(sample)
            label = sample[self.label_col]
            if predict == label and label == 1:
                t_p += 1
            elif predict == label and label == 0:
                t_n += 1
            elif predict != label and predict == 1:
                f_p += 1
            else:
                f_n += 1
        return t_p / p_total, f_p / n_total, f_n / p_total, t_n / n_total

    @staticmethod
    def k_cv(data, k, n_trees=10, min_sample_split=2, n_features='sqrt', criterion='gini'):
        labels = list(set(data[data.columns[0]].tolist()))
        cv_scores = []
        for i in range(0, k):
            print('%d validation begin' % (i + 1))
            cv_test_list = []
            cv_train_list = []
            for label in labels:
                data_label = data[data[data.columns[0]] == label]
                n_fold = int(data_label.shape[0] / k)
                cv_test_list.append(data_label[n_fold * i: n_fold * (i + 1)])
                cv_train_list.append(pd.concat([data_label[: n_fold * i], data_label[n_fold * (i + 1):]]))
            cv_train = pd.concat(cv_train_list)
            cv_train.reset_index(drop=True, inplace=True)
            cv_test = pd.concat(cv_test_list)
            cv_test.reset_index(drop=True, inplace=True)
            cv_rf = RandomForest(n_trees=n_trees,
                                 min_sample_split=min_sample_split,
                                 n_features=n_features,
                                 criterion=criterion)
            cv_rf.fit(cv_train)
            cv_scores.append(cv_rf.score(cv_test))
            print('validation score: %f' % cv_scores[i])
        return cv_scores

    # params [(n_trees1, min_sample_split1, n_features1, criterion1), (n_trees2, min_sample_split2, n_features2, criterion2)...]
    # return [[(n_trees1, min_sample_split1, n_features1, criterion1), mean, (interval)]...]
    @staticmethod
    def plot(data, k, params):
        results = []
        for param in params:
            cv_scores = RandomForest.k_cv(data, k, param[0], param[1], param[2], param[3])
            mean = np.mean(cv_scores)
            interval = st.norm.interval(0.95, loc=mean, scale=st.sem(cv_scores))
            result = [param, mean, interval]
            results.append(result)
        return results
