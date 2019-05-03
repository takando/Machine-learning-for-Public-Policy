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

threshold = 0.20


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

dic_lst = []
metric_lst = ["accuracy", "precision", "recall", "f1", "ap", "auc"]
for data_set in data_sets:
    train_x, test_x, train_y, test_y = data_set
    rfmodel = revmlpipeline.random_forest(train_x, train_y)
    bagmodel = revmlpipeline.bagging(train_x, train_y)
    kmodel = revmlpipeline.k_neighbors(4, train_x, train_y)
    treemodel = revmlpipeline.decision_tree(train_x, train_y)
    svmmodel = revmlpipeline.svm(train_x, train_y)
    boostmodel = revmlpipeline.boosting(train_x, train_y)
    logmodel = revmlpipeline.logistic_regression(train_x, train_y)
    model_lst = [kmodel, treemodel, svmmodel, logmodel, rfmodel, bagmodel, boostmodel]
    metric_dic = {}
    for model in model_lst:
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
