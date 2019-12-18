import pandas as pd
import numpy as np
import scipy.stats as st
import math
import random


class Node:
    def __init__(self, data_index, depth=None, left=None, right=None, feature=None, boundary=None, label=None):
        self.data_index = data_index
        self.depth = depth
        self.left = left
        self.right = right
        self.feature = feature
        self.boundary = boundary
        self.label = label


class DecisionTree:
    root = None
    min_sample_split = 2
    max_depth = None
    n_features = None
    criterion = 'gini'  # 'entropy': information gain; 'ratio': information gain ratio; 'gini': gini; default: 'gini'
    train_data = None
    label_col = None
    depth = 0

    def __init__(self, min_sample_split=2, max_depth=None, n_features=None, criterion='gini'):
        self.root = None
        self.train_data = None
        self.label_col = None
        self.depth = 0
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.criterion = criterion

    def get_entropy(self, data):
        counts = data[self.label_col].value_counts().values
        n_total = sum(counts)
        entropy = 0.0
        for count in counts:
            p = count / n_total
            # entropy -= p * math.log(p, 2)
            entropy -= p * np.log2(p)
        return entropy

    def get_information_gain(self, total, left, right):
        total_entropy = self.get_entropy(total)
        left_entropy = self.get_entropy(left)
        right_entropy = self.get_entropy(right)
        child_entropy = (left.shape[0] * left_entropy + right.shape[0] * right_entropy) / total.shape[0]
        return total_entropy - child_entropy

    def get_gain_ratio(self, total, left, right):
        left_p = left.shape[0] / total.shape[0]
        right_p = right.shape[0] / total.shape[0]
        # split_information = 0 - (left_p * math.log(left_p, 2) + right_p * math.log(right_p, 2))
        split_information = 0 - (left_p * np.log2(left_p) + right_p * np.log2(right_p))
        information_gian = self.get_information_gain(total, left, right)
        return information_gian / split_information

    def get_gini(self, data):
        counts = data[self.label_col].value_counts().values
        n_total = sum(counts)
        ps = [count / n_total for count in counts]
        # return 1 - sum(math.pow(p, 2) for p in ps)
        return 1 - sum(np.square(p) for p in ps)

    def get_gain_gini(self, total, left, right):
        left_gini = self.get_gini(left)
        right_gini = self.get_gini(right)
        gini = (left.shape[0] * left_gini + right.shape[0] * right_gini) / total.shape[0]
        return 2 - gini  # gini less is better

    def get_gain(self, total, left, right):
        if self.criterion == 'entropy':
            return self.get_information_gain(total, left, right)
        elif self.criterion == 'gini':
            return self.get_gain_gini(total, left, right)
        else:
            return self.get_gain_ratio(total, left, right)

    def get_sub_features(self):
        features = self.train_data.columns.tolist()[1:]
        random.shuffle(features)
        if self.n_features is None or self.n_features == 'all':
            return features
        elif self.n_features == 'sqrt':
            # n_sub_features = int(math.sqrt(len(features)))
            n_sub_features = int(np.sqrt(len(features)))
            return features[:n_sub_features]
        elif self.n_features == 'log':
            # n_sub_features = int(math.log(len(features), 2))
            n_sub_features = int(np.log2(len(features)))
            return features[:n_sub_features]
        else:
            n_sub_features = int(len(features) * self.n_features)
            return features[:n_sub_features]

    def get_best_division(self, data):
        # features = self.train_data.columns.tolist()[1:]
        features = self.get_sub_features()
        best_fbg = None  # (feature, boundary, gain)
        for feature in features:
            values = sorted(data[feature].tolist())
            boundaries = []
            for i in range(0, len(values) - 1):
                if values[i] != values[i + 1]:
                    boundaries.append((values[i] + values[i + 1]) / 2)
            for boundary in boundaries:
                left_data = data[data[feature] <= boundary]
                right_data = data[data[feature] > boundary]
                gain = self.get_gain(data, left_data, right_data)
                if best_fbg is None or gain > best_fbg[2]:
                    best_fbg = (feature, boundary, gain)
        # information gain should be bigger than 0?
        if best_fbg is not None and best_fbg[2] > 0:
            return best_fbg[0], best_fbg[1]  # (feature, boundary)
        else:
            return None

    def cal_label(self, data):
        return data[self.label_col].mode()[0]

    def divide(self, root):
        data = self.train_data.iloc[root.data_index, :]
        if root.depth == self.max_depth or len(root.data_index) < self.min_sample_split:
            root.label = self.cal_label(data)
            return None

        best_division = self.get_best_division(data)
        if best_division is None:
            root.label = self.cal_label(data)
            return None

        root.feature = best_division[0]
        root.boundary = best_division[1]
        left_data_index = data[data[root.feature] <= root.boundary].index
        right_data_index = data[data[root.feature] > root.boundary].index
        root.left = Node(data_index=left_data_index, depth=root.depth + 1)
        root.right = Node(data_index=right_data_index, depth=root.depth + 1)
        return root.left, root.right

    def build(self):
        tree = [self.root]
        i = 0
        j = 0
        while True:
            branches = self.divide(tree[i])
            if branches is not None:
                tree.extend(branches)
                j += 2
            if i == j:
                break
            else:
                i += 1
        self.depth = max(node.depth for node in tree)

    def fit(self, train_data):
        self.train_data = train_data
        self.label_col = train_data.columns[0]
        self.root = Node(data_index=train_data.index, depth=1)
        self.build()

    def predict_one(self, sample):
        if self.root is None:
            print("Classifier should be fit first!")
            return None
        node = self.root
        while True:
            if node.label is not None:
                return node.label
            elif sample[node.feature] <= node.boundary:
                node = node.left
            else:
                node = node.right

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

    @staticmethod
    def k_cv(data, k, min_sample_split=2, max_depth=None, n_features=None, criterion='gini'):
        labels = list(set(data[data.columns[0]].tolist()))
        # n_fold = int(data.shape[0] / k)
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

            cv_dt = DecisionTree(min_sample_split=min_sample_split,
                                 max_depth=max_depth,
                                 n_features=n_features,
                                 criterion=criterion)
            cv_dt.fit(cv_train)
            cv_scores.append(cv_dt.score(cv_test))
            print('validation score: %f' % cv_scores[i])
        return cv_scores

    # params [(min_sample_split1, max_depth1, n_features1, criterion1), (min_sample_split2, max_depth2, n_features2, criterion2)...]
    # return [[(min_sample_split1, max_depth1, n_features1, criterion1), mean, (interval)]...]
    @staticmethod
    def plot(data, k, params):
        results = []
        for param in params:
            cv_scores = DecisionTree.k_cv(data, k, param[0], param[1], param[2], param[3])
            mean = np.mean(cv_scores)
            interval = st.norm.interval(0.95, loc=mean, scale=st.sem(cv_scores))
            result = [param, mean, interval]
            results.append(result)
        return results
