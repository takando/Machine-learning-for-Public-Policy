import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline


def read_data(file_name):
	return pd.read_csv("credit-data.csv")


class explorer:
    def __init__(self, df):
        self.df = df
        
    def mean(self, attr):
        return np.nanmean(self.df[attr])
    
    def std(self, attr):
        return np.nanstd(self.df[attr])
        
    def find_outlier(self, attr):
        data = self.df
        array = data[attr]
        q1, q3= np.nanpercentile(array,[25,75])
        iqr = q3 - q1
        lower_bound = q1 -(1.5 * iqr)
        upper_bound = q3 +(1.5 * iqr)
        outlier = np.array(array > upper_bound)
        return data[outlier]
