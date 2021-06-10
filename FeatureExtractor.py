import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

from ClassifiersOptimizer import ClassifiersOptimizer
from DataSplitter import Splitter
import ClassifiersConfig

class FeatureExtractor():
    def __init__(self, data):        
        self.__data = data
        self.__splitter = Splitter(data)
        self.__train_x, self.__train_y, self.__test_x, self.__test_y = self.__splitter.get_splited_data()
        
        self.__optimizer = ClassifiersOptimizer({'ExtraTreesClassifier': ExtraTreesClassifier()}, data)
        
        self.__cls_params = None
        self.__extra_tree_forest = None
        self.__feature_importance = None
        self.__feature_importance_normalized = None
        self.__extra_tree_forest = None
        
    def tune(self, key, params):
        self.__cls_params = self.__optimizer.run_randomized_search_cv(key, params)
        self.__extra_tree_forest = ExtraTreesClassifier(**self.__cls_params)
                
    def extract(self):
        if(self.__extra_tree_forest is None):
            self.tune('ExtraTreesClassifier', ClassifiersConfig.etc_params_to_tune)
            
#         # Building the model
        self.__extra_tree_forest = ExtraTreesClassifier(**self.__cls_params)

#         # Training the model
        self.__extra_tree_forest.fit(self.__train_x, self.__train_y.ravel())

        # Computing the importance of each feature
        self.__feature_importance = self.__extra_tree_forest.feature_importances_

        print('Features importance: \n', self.__feature_importance)
        
        # Normalizing the individual importances
        self.__feature_importance_normalized = np.std([tree.feature_importances_ for tree in self.__extra_tree_forest.estimators_], axis = 0)
        
        self.plot_importance()
        
        return self.__feature_importance_normalized
        
    def plot_importance(self):
        # Plotting a Bar Graph to compare the models

        features_size = len(self.__train_x.columns)
        axes_size = int(features_size/5)
        if features_size%5 != 0:
            axes_size += 1

        # plt.figure(figsize=(20, 8), dpi=80)
        fig, axs = plt.subplots(axes_size, figsize=(20, 24), dpi=80)

        indx = 0
        for i in range(0, features_size, 5):
            if i < len(self.__test_x.columns):
                axs[indx].bar(self.__test_x.columns[i:i+5], self.__feature_importance_normalized[i:i+5])
            else:
                axs[indx].bar(self.__test_x.columns[i:features_size], self.__feature_importance_normalized[i:features_size])
            indx += 1

        fig.suptitle('Comparison of different Feature Importances', fontsize=20)
        fig.text(0.5, 0.04, 'Feature Labels', ha='center', fontsize=20)
        fig.text(0.04, 0.5, 'Feature Importances', va='center', rotation='vertical', fontsize=20)
        fig.show()