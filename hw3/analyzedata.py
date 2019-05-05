import revmlpipeline
from IPython.display import display
import warnings

warnings.filterwarnings('ignore')


file_path = "projects_2012_2013.csv"

col_to_transform = ["school_metro", "poverty_level", "eligible_double_your_impact_match"]

col_name = "date_posted"

span = 6

start_date = "01/01/2012"

x_cols = ['poverty_level_highest poverty', 'poverty_level_low poverty',
       'poverty_level_moderate poverty','students_reached',
       'eligible_double_your_impact_match_t']

y_col = "not_within_60days"


df = revmlpipeline.read_data(file_path)

df = df.dropna(how="any")

df = pd.get_dummies(df, dummy_na=False, columns = col_to_transform, drop_first=True)

date = lambda x: datetime.datetime.strptime(x, '%m/%d/%y')
df["date_posted"] = df["date_posted"].map(date)
df["datefullyfunded"] = df["datefullyfunded"].map(date)

after_60days = datetime.timedelta(days=60)
df["not_within_60days"] = df["datefullyfunded"] > df["date_posted"] + after_60days
df["not_within_60days"] = df["not_within_60days"].astype(int)

data_sets = []
for leng in [1, 2, 3]:
    split_data = revmlpipeline.time_split(df, span, leng, start_date, col_name, x_cols, y_col)
    data_sets.append(split_data)

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
    print()
