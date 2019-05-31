import revmlpipeline
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve,\
average_precision_score, roc_curve, roc_auc_score
from IPython.display import display
import warnings

warnings.filterwarnings('ignore')

file_path = "projects_2012_2013.csv"

col_to_transform = ["teacher_prefix", "school_metro", "poverty_level", "eligible_double_your_impact_match"]

col_name = "date_posted"

train_span = 6 #duration of training data

test_span = 4 #duration of test data, which is shorter than training data because of leaving 2 month gap between
              #training and test data  

start_date = "01/01/2012"

x_cols = ["teacher_prefix_Mrs.", "teacher_prefix_Ms.", "students_reached", "school_metro_suburban", "school_metro_urban", "poverty_level_highest poverty", "poverty_level_low poverty", "poverty_level_moderate poverty",
"eligible_double_your_impact_match_t"]

y_col = "not_within_60days"

percentile = 0.95 #the value above which project is deemed as high risk of failing

df = revmlpipeline.read_data(file_path)
df = df.dropna(how="any")

#Make dummy label which indicates the project failed to get funds in 60days or not 
date = lambda x: datetime.datetime.strptime(x, '%m/%d/%y')
df["date_posted"] = df["date_posted"].map(date)
df["datefullyfunded"] = df["datefullyfunded"].map(date)

after_60days = datetime.timedelta(days=60)
df["not_within_60days"] = df["datefullyfunded"] > df["date_posted"] + after_60days
df["not_within_60days"] = df["not_within_60days"].astype(int)

#split data into training and test data
data_sets = []
for leng in [1, 2, 3]:
    split_data = revmlpipeline.time_split(df, train_span, test_span, leng, start_date, col_name, y_col)
    data_sets.append(split_data)

#Make dummy variable for features
for data_set in data_sets:
    train_x = data_set[0]
    test_x = data_set[1]
    train_x = pd.get_dummies(train_x, dummy_na=False, columns = col_to_transform, drop_first=True)
    test_x = pd.get_dummies(test_x, dummy_na=False, columns = col_to_transform, drop_first=True)
    for col in x_cols:
        if col not in train_x.columns:
            train_x[col] = 0    
        if col not in test_x.columns:
            test_x[col] = 0
    train_x = train_x[x_cols]
    test_x = test_x[x_cols]
    data_set[0] = train_x
    data_set[1] = test_x

def experiment(clf=None):
    """
    Experiment different models with different hyper parameters.
    """
    if clf == "KNN":
        for num_neighbors in [1,3,10,20,50]:
            metr_lst = []
            for dataset in data_sets:
                train_x, test_x, train_y, test_y = dataset
                model = revmlpipeline.k_neighbors(train_x, train_y, num_neighbors)
                metrics = revmlpipeline.evaluate_model(model, test_x, test_y, 0.3)
                metr_lst.append(metrics)
            metr_array = np.array(metr_lst)
            print("KNN; k=" + str(num_neighbors))
            print("accuracy, precision, recall, f1, ap, auc")
            print(np.sum(metr_array, axis=0)/3)
            

    elif clf == "decision_tree":
        for depth in [1,3,5,10,20]:
            metr_lst = []
            for dataset in data_sets:
                train_x, test_x, train_y, test_y = dataset
                model = revmlpipeline.decision_tree(train_x, train_y, depth)
                metrics = revmlpipeline.evaluate_model(model, test_x, test_y, 0.3)
                metr_lst.append(metrics)
            metr_array = np.array(metr_lst)
            print("DecisionTree; depth=" + str(depth))
            print("accuracy, precision, recall, f1, ap, auc")
            print(np.sum(metr_array, axis=0)/3)
            

    elif clf == "SVM":
        for c in [0.1, 1, 10, 100]:
            metr_lst = []
            for dataset in data_sets:
                train_x, test_x, train_y, test_y = dataset
                model = revmlpipeline.lsvm(train_x, train_y, "l2", c)
                metrics = revmlpipeline.evaluate_model(model, test_x, test_y, 0.3)
                metr_lst.append(metrics)
            metr_array = np.array(metr_lst)
            print("SVM; C=" + str(c))
            print("accuracy, precision, recall, f1, ap, auc")
            print(np.sum(metr_array, axis=0)/3)


    elif clf == "logistic_regression":
        for c in [0.1, 1, 10, 100]:
            metr_lst = []
            for dataset in data_sets:
                train_x, test_x, train_y, test_y = dataset
                model = revmlpipeline.logistic_regression(train_x, train_y, "l2", c)
                metrics = revmlpipeline.evaluate_model(model, test_x, test_y, 0.3)
                metr_lst.append(metrics)
            metr_array = np.array(metr_lst)
            print("Log; C=" + str(c))
            print("accuracy, precision, recall, f1, ap, auc")
            print(np.sum(metr_array, axis=0)/3)
            

    elif clf =="random_forest":
        for n in [1, 5, 10, 20, 50, 100]:
            metr_lst = []
            for dataset in data_sets:
                train_x, test_x, train_y, test_y = dataset
                model = revmlpipeline.random_forest(train_x, train_y, n)
                metrics = revmlpipeline.evaluate_model(model, test_x, test_y, 0.3)
                metr_lst.append(metrics)
            metr_array = np.array(metr_lst)
            print("RF; n_estimator=" + str(n))
            print("accuracy, precision, recall, f1, ap, auc")
            print(np.sum(metr_array, axis=0)/3)


def analyze():
    """
    Calculate precision, recall for different thresholds, along with auc
    """
    table_lst_perset = []
    for data_set in data_sets:
        train_x, test_x, train_y, test_y = data_set
        rfmodel = revmlpipeline.random_forest(train_x, train_y, 20)
        bagmodel = revmlpipeline.bagging(train_x, train_y)
        kmodel = revmlpipeline.k_neighbors(train_x, train_y, 4)
        treemodel = revmlpipeline.decision_tree(train_x, train_y, None)
        svmmodel = revmlpipeline.lsvm(train_x, train_y, "l2", 1.0)
        boostmodel = revmlpipeline.boosting(train_x, train_y)
        logmodel = revmlpipeline.logistic_regression(train_x, train_y, "l2", 1.0)
        baseline = "baseline"
        model_lst = [baseline, kmodel, treemodel, svmmodel, logmodel, rfmodel, bagmodel, boostmodel]
        table_lst = []
        for model in model_lst:
            if model == baseline:
                index_lst = ["precision", "recall"]
            else:
                index_lst = ["precision", "recall", "auc"]
            table_dic = {}
            for threshold in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
                if model == baseline:
                    pred_label = [1] * test_y.size
                    precision = precision_score(test_y, pred_label)
                    recall = recall_score(test_y, pred_label)
                    table_dic[threshold] = [precision, recall]
                else:
                    accuracy, precision, recall, f1, ap, auc = revmlpipeline.evaluate_model(model, test_x, test_y, threshold)
                    table_dic[threshold] = [precision, recall, auc]
            model_table = pd.DataFrame(data = table_dic, index=index_lst)
            table_lst.append(model_table)
        table_lst_perset.append(table_lst)

    agg_lst = []
    table1, table2, table3 = table_lst_perset
    for model_table1, model_table2, model_table3 in zip(table1, table2, table3):
        model_table4 = (model_table1 + model_table2 + model_table3)/3
        agg_lst.append(model_table4)

    model_name = ["Baseline", "KNN model", "Decision Tree model", "SVM model", "Logistic regression model", "Random forest model", "Bagging model", "Boosting model"]
    for table, name in zip(agg_lst, model_name):
        print(name + ";")
        display(table)


def compare_models():
    """
    Compare the performance of models across all metrics. 
    For evaluation, we find out thresholds according to the population of probability.
    """
    dic_lst = []
    metric_lst = ["accuracy", "precision", "recall", "f1", "ap", "auc"]
    for data_set in data_sets:
        train_x, test_x, train_y, test_y = data_set
        rfmodel = revmlpipeline.random_forest(train_x, train_y, 20)
        bagmodel = revmlpipeline.bagging(train_x, train_y)
        kmodel = revmlpipeline.k_neighbors(train_x, train_y, 4)
        treemodel = revmlpipeline.decision_tree(train_x, train_y, None)
        svmmodel = revmlpipeline.lsvm(train_x, train_y, "l2", 1.0)
        boostmodel = revmlpipeline.boosting(train_x, train_y)
        logmodel = revmlpipeline.logistic_regression(train_x, train_y, "l2", 1.0)
        model_lst = [kmodel, treemodel, svmmodel, logmodel, rfmodel, bagmodel, boostmodel]
        metric_dic = {}
        for model in model_lst:
            if str(type(model)) == "<class 'sklearn.svm.classes.LinearSVC'>":
                prob_pos = model.decision_function(test_x)
                prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
                pred_prob = np.array([1-x for x in prob_pos])
            else:
                pred_scores = model.predict_proba(test_x)
                pred_prob = pred_scores[:, 1]
            threshold = np.quantile(pred_prob, percentile) #find out threshold based on the distribution of probability
            accuracy, precision, recall, f1, ap, auc = revmlpipeline.evaluate_model(model, test_x, test_y, threshold)
            for metric, name in zip([accuracy, precision, recall, f1, ap, auc], metric_lst):
                if not name in metric_dic.keys():
                    metric_dic[name] = [metric]
                else:
                    metric_dic[name].append(metric)
        dic_lst.append(metric_dic)

    dic4 = {}
    dic1, dic2, dic3 = dic_lst
    for name in metric_lst:
        dic4[name] = (np.array(dic1[name]) + np.array(dic2[name]) + np.array(dic3[name]))/3 

    X = [(0, 0, 0),(1, 0, 1), (2, 0, 2), (3, 1, 0), (4, 1, 1), (5, 1, 2)]
    label = ["KNN", "Tree", "SVM", "Log", "RF", "Bagg", "Boost"]
    left = np.array([1,2,3,4,5,6,7])

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(17,10))
    for index, name in enumerate(metric_lst):
        ind, row, col = X[index]
        height = dic4[name]
        axes[row, col].bar(left, height, tick_label=label)
        axes[row, col].set_title(name)
    plt.show()