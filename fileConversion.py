import pandas as pd
import numpy as np
import random


class FileConversion:

    def __init__(self, file_path):
        self._dataFrame = pd.read_csv('dataset/'+file_path)
        self._savePath = 'finalDataset2/'+file_path

    def get_head(self):
        return self._dataFrame.head()

    def replacer(self):
        condition = {
            'Customer Type': {
                'disloyal Customer': 0,
                'Loyal Customer': 1
            },
            'Type of Travel': {
                'Personal Travel': 0,
                'Business travel': 1,
            },
            'Class': {
                'Eco': 0,
                'Eco Plus': 1,
                'Business': 2
            },
            'satisfaction': {
                'neutral or dissatisfied': 0,
                'satisfied': 1
            }
        }

        self._dataFrame = self._dataFrame.replace(condition)
        self._dataFrame = self._dataFrame.drop(columns=['record'])

    def gender_splitter(self):
        self._dataFrame.loc[self._dataFrame['Gender'] == 'Male', 'Male'] = 1
        self._dataFrame.loc[self._dataFrame['Gender'] != 'Male', 'Male'] = 0
        self._dataFrame.loc[self._dataFrame['Gender'] == 'Female', 'Female'] = 1
        self._dataFrame.loc[self._dataFrame['Gender'] != 'Female', 'Female'] = 0
        self._dataFrame = self._dataFrame.drop(columns=['Gender'])

    def save_csv(self):
        self._dataFrame.to_csv(self._savePath, index=False)

    def add_missing_values(self):
        fractal = 0.1/len(self._dataFrame.columns)
        for col in self._dataFrame.columns:
            self._dataFrame.loc[self._dataFrame.sample(frac=fractal).index, col] = np.nan


if __name__ == "__main__":
    for file in ['test.csv', 'train.csv']:
        fc = FileConversion(file)
        fc.replacer()
        fc.gender_splitter()
        fc.add_missing_values()
        fc.save_csv()
        del fc
