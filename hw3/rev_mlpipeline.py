import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


def read_data(file_path):
    return pd.read_csv(file_path)


def describe(df):
    return df.describe()
        

# def find_outlier(df, attr):
#     array = df[attr]
#     q1, q3= np.nanpercentile(array,[25,75])
#     iqr = q3 - q1
#     lower_bound = q1 -(1.5 * iqr)
#     upper_bound = q3 +(1.5 * iqr)
#     outlier = np.array(array > upper_bound)
#     return df[outlier]


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


def build_classifier(df, drop_vars, target_label):
    X = df.drop(drop_vars, axis=1)
    Y = df[target_label]
    Y = np.ravel(Y)
    (x_train, x_test, y_train, y_test) = train_test_split(X, Y)
    log_model = LogisticRegression(random_state=0) 
    return log_model.fit(x_train, y_train), x_test, y_test


def evaluate_model(model, x_test, y_test):
    class_predict = model.predict(x_test) 
    return metrics.accuracy_score(y_test, class_predict)  
