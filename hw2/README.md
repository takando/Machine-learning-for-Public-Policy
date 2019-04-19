# Machine learning homework2

Takuya Ando

## Files

-hw2jup-Copy1.ipynb  
Writeup file that shows results by running functions in mlpipeline file.

-mlpipeline.py  
Includes all machine learning pipeline components for this assignment

-usepipeline.py  
Execute mlpipeline.py file and show the result of model evaluation, as well as plots for variables

## ML pipleline functions

-read_data(file_path)  
Read csv data into pandas dataframe

-describe(df)  
Privide summary statistics of dataframe

-plot(X, df, drop_lst)  
Show histograms of variables

-pre_process(df)  
Fill missing values with mean of the variable

-descretize(df, attr, col_name, num_bin, label_lst)  
Descritize continuous variables

-make_dummy(df, col_lst)  
Convert categorical values into binary variables

-build_classifier(df, drop_vars, target_label)  
Build ML model(use logistic regression for this assignment)

-evaluate_model(model, x_test, y_test)  
Compute accuracy of the model

## How to run

Run codes below in jupyter notebook

%run -i mlpipeline.py  
%run -i usepipeline.py