import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve,\
average_precision_score, roc_curve, roc_auc_score 


def read_data(file_path):
    return pd.read_csv(file_path)


def describe(df):
    return df.describe()


def plot(X, df, drop_lst):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(17,20))
    for x in X:
        ind, row, col = x
        attr = df.drop(drop_lst, axis=1).columns[ind]
        if attr == 'age' or attr == 'NumberOfOpenCreditLinesAndLoans':
            axes[row, col].hist(df[attr], bins=10)
        else:
            axes[row, col].hist(df[attr], bins=10, log=True)
        axes[row, col].set_title(attr)
    plt.show()


def pre_process(df):
    fill_dic = {}
    for i, val in df.isnull().any().items():
        if val == True:
            fill_dic[i] = np.nanmean(df[i])
    return df.fillna(fill_dic)


def descretize(df, attr, col_name, num_bin, label_lst):
    df2 = pd.qcut(df[attr], num_bin, labels=label_lst)
    df[col_name] = df2
    return df


def make_dummy(df, col_lst):
    df = pd.get_dummies(df, dummy_na=False, drop_first=True, columns = col_lst)
    return df


def data_split(df, drop_vars, target_label):
    X = df.drop(drop_vars, axis=1)
    Y = df[target_label]
    Y = np.ravel(Y)
    (x_train, x_test, y_train, y_test) = train_test_split(X, Y)
    return (x_train, x_test, y_train, y_test)


def time_split(df, train_span, test_span, length, start_time, col_name, y_col):
    train_start = datetime.datetime.strptime(start_time, '%m/%d/%Y')
    train_end = train_start + relativedelta(months=train_span*length)
    test_start = train_end + relativedelta(days=60)
    test_end = test_start + relativedelta(months=test_span)
    filter_train1 = train_start <= df[col_name]
    filter_train2 = df[col_name] <= train_end
    filter_test1 = test_start <= df[col_name]
    filter_test2 = df[col_name] <= test_end
    train_data = df[filter_train1][filter_train2]
    train_x = train_data[train_data.columns[train_data.columns!=y_col]]
    train_y = train_data[y_col]
    test_data = df[filter_test1][filter_test2]
    test_x = test_data[train_data.columns[train_data.columns!=y_col]]
    test_y = test_data[y_col]
    return [train_x, test_x, train_y, test_y]


def k_neighbors(x_train, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski')
    return knn.fit(x_train, y_train)


def decision_tree(x_train, y_train, depth):
    dec_tree = DecisionTreeClassifier(random_state=0, max_depth=depth) 
    return dec_tree.fit(x_train, y_train)


def lsvm(x_train, y_train, penalty, c):
    lsvm = LinearSVC(random_state=0, penalty=penalty, C=c)
    return lsvm.fit(x_train, y_train)


def logistic_regression(x_train, y_train, penalty, c):
    log_model = LogisticRegression(random_state=0, penalty=penalty, C=c) 
    return log_model.fit(x_train, y_train)


def random_forest(x_train, y_train, n):
    r_for = RandomForestClassifier(n_estimators=n, random_state=0)
    return r_for.fit(x_train, y_train)


def bagging(x_train, y_train):
    bagg = BaggingClassifier(LogisticRegression(random_state=0), max_samples=0.5, max_features=0.5) 
    return bagg.fit(x_train, y_train)


def boosting(x_train, y_train):
    boost = AdaBoostClassifier(n_estimators=20)
    return boost.fit(x_train, y_train)


def evaluate_model(model, x_test, y_test, threshold):
    if str(type(model)) == "<class 'sklearn.svm.classes.LinearSVC'>":
        prob_pos = model.decision_function(x_test)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        pred_prob = [1-x for x in prob_pos]
        pred_label = [1 if (1-x) >threshold else 0 for x in pred_prob]
    else:
        pred_scores = model.predict_proba(x_test)
        pred_prob = pred_scores[:, 1]
        pred_label = [1 if x[1]>threshold else 0 for x in pred_scores]
    accuracy = accuracy_score(y_test, pred_label)
    precision = precision_score(y_test, pred_label)
    recall = recall_score(y_test, pred_label)
    f1 = f1_score(y_test, pred_label)
    ap = average_precision_score(y_test, pred_prob)
    auc = roc_auc_score(y_test, pred_prob)
    return accuracy, precision, recall, f1, ap, auc
