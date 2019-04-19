import mlpipeline

file_path = "credit-data.csv"

drop_lst = ["PersonID","SeriousDlqin2yrs","zipcode"]

X = [(0, 0, 0),(1, 0, 1), (2, 0, 2), (3, 1, 0), (4, 1, 1), (5, 1, 2), (6, 2, 0), (7, 2, 1), (8, 2, 2), (9, 3, 0)]

attr = "age"

num_bin = 3

label_lst = ["young", "middle", "elderly"]

col_name = "age_cat"

col_lst = ["age_cat"]

drop_vars = ["PersonID", "SeriousDlqin2yrs", "age", "zipcode", "age_cat_middle", "age_cat_elderly"]

target_label = "SeriousDlqin2yrs"



df = mlpipeline.read_data(file_path)

print(describe(df))

mlpipeline.plot(X, df, drop_lst)

df = mlpipeline.pre_process(df)

df = mlpipeline.descretize(df, attr, col_name, num_bin, label_lst)

df = mlpipeline.make_dummy(df, col_lst)

(log_model, x_test, y_test) = mlpipeline.build_classifier(df, drop_vars, target_label)

print("model_accuracy:" + str(evaluate_model(log_model, x_test, y_test)))