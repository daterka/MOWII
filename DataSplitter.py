from numpy.random import RandomState
import pandas as pd
import numpy as np

class Splitter:
    def __init__(self, dataset, split=0.2):
        self.__df = dataset
        if 'ind' in self.__df.columns:
            self.__df = self.__df.drop(['ind'], axis=1)
        self.__df.reset_index(drop=True, inplace=True)
        self.__train = self.__df.sample(frac=1-split, random_state=RandomState())
        self.__test = self.test_data()
        self.__x = self.dataset_X()
        self.__y = self.dataset_y()
        self.__train_x = None
        self.__train_y = None
        self.__test_x = None
        self.__test_y = None
        self.split_train_test()
        
    def dataset_X(self):
        df = self.__df.drop(columns=['satisfaction'])
        df.reset_index(inplace=True)
        df.drop(['index'], axis=1, inplace=True)
        return df
    
    def dataset_y(self):
        df = self.__df['satisfaction']
        df.reset_index(drop=True, inplace=True)
        return df
        
    def train_data(self):
        df = self.__train
        df.reset_index(inplace=True)
        df.drop(['index'], axis=1, inplace=True)
        return df
    
    def test_data(self):
        df = self.__df.loc[~self.__df.index.isin(self.__train.index)]
        df.reset_index(drop=True, inplace=True)
        return df
    
    def train_X_data(self):
        df = self.__train
        df = df.drop(columns=['satisfaction'])
        df.reset_index(inplace=True)
        df.drop(['index'], axis=1, inplace=True)
        return df
    
    def train_Y_data(self):
        df = self.__train['satisfaction']
        df.reset_index(drop=True, inplace=True)
#         df.drop(['index'], axis=1, inplace=True)
        return df
    
    def test_X_data(self):
        df = self.__df.loc[~self.__df.index.isin(self.__train.index)]
        df = df.drop(columns=['satisfaction'])
        df.reset_index(inplace=True)
        df.drop(['index'], axis=1, inplace=True)
        return df
    
    def test_Y_data(self):
        df = self.__df.loc[~self.__df.index.isin(self.__train.index)]['satisfaction']
        df.reset_index(drop=True, inplace=True)
#         df.drop(['index'], axis=1, inplace=True)
        return df
    
    def split_train_test(self):
        self.__train_x = self.train_X_data()
        self.__train_y = self.train_Y_data()
        self.__test_x = self.test_X_data()
        self.__test_y = self.test_Y_data()
        return self.__train_x, self.__train_y, self.__test_x, self.__test_y
    
    def get_splited_data(self):
        if self.__train_x is None:
            return self.split_train_test()
        return self.__train_x, self.__train_y, self.__test_x, self.__test_y
    
    def get_data(self):
        return self._df
    
    def get_train_x(self):
        if self.__train_x is None:
            self.split_train_test()
        return self.__train_x
    
    def get_train_y(self):
        if self.__train_y is None:
            self.split_train_test()
        return self.__train_y
    
    def get_test_x(self):
        if self.__test_x is None:
            self.split_train_test()
        return self.__test_x
    
    def get_test_y(self):
        if self.__test_y is None:
            self.split_train_test()
        return self.__test_y