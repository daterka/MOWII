from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingClassifier
from itertools import product
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt

from DataSplitter import Splitter

import ClassifiersConfig

class ClassifiersOptimizer:
    def __init__(self, classifiers, data):
        self.__classifiers = classifiers
        
        self.__data = data
        self.__splitter = Splitter(data)
        self.__train_x, self.__train_y, self.__test_x, self.__test_y = self.__splitter.get_splited_data()
                
        self.__grid_search_cv = None    
        self.__randomized_search_cv = None   
            
        self.__picked_classifier = None
        self.__stats = None
        self.__conf_mat = None
            
    def perform_voting(self, vtype='hard', weights=None, verbose=True):
        labels = [x for x in self.__classifiers.keys()]
        labels.append('Ensamble-'+vtype)
        labels
        
        clfs = [x for x in self.__classifiers.values()]
        eclfs = [(k, v) for k, v in self.__classifiers.items()]
        vclf = None
        
        if vtype == 'hard':
            vclf = VotingClassifier(estimators=eclfs, voting='hard', verbose=1)
        else:
            vclf = VotingClassifier(estimators=clfs, voting='soft', verbose=1)
                                    
        clfs.append(vclf)
        
        for clf, label in zip(clfs, labels):
            scores = cross_val_score(clf, self.__train_x, self.__train_y, scoring='accuracy', cv=5, n_jobs=-1)
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
            self.plot_conf_mat(clf, label)
            
        self.print_decision_boundaries(clfs, labels)
            
    def perform_stacking(self):
        eclfs = [(k, v) for k, v in self.__classifiers.items()]
        clf = StackingClassifier(estimators=eclfs, final_estimator=LogisticRegression(), cv=5, verbose=1, n_jobs=-1)
        clf.fit(self.__train_x, self.__train_y)
        scores = clf.score(self.__test_x, self.__test_y)
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), 'StackingClassifier'))
        self.plot_conf_mat(clf, 'StackingClassifier')
        return clf
        
    def run_grid_searach_cv(self, key, params):
        clf = self.__classifiers[key]
        self.__grid_search_cv = GridSearchCV(estimator=clf, param_grid=params, cv=3, verbose=1, n_jobs=-1)
        self.__grid_search_cv.fit(self.__train_x, self.__train_y.ravel())
        print('CV results: \n', sorted(self.__grid_search_cv.cv_results_))
        print('Best params: \n', self.__grid_search_cv.best_params_)
        print('Accuracy: ', self.__grid_search_cv.best_score_)
        self.plot_conf_mat(self.__grid_search_cv.best_estimator_, key)
        return self.__grid_search_cv.best_params_
    
    def run_randomized_search_cv(self, key, params):
        self.__randomized_search_cv = RandomizedSearchCV(RandomForestClassifier(), params, n_iter=10, cv=5, verbose=1, n_jobs=-1)
        self.__randomized_search_cv.fit(self.__train_x, self.__train_y.ravel())
        print('CV results: \n', sorted(self.__randomized_search_cv.cv_results_))
        print('Best params: \n', self.__randomized_search_cv.best_params_)
        print('Accuracy: ', self.__randomized_search_cv.best_score_)
        self.plot_conf_mat(self.__randomized_search_cv.best_estimator_, key)
        return self.__randomized_search_cv.best_params_
    
    def plot_decision_boundaries(self, clfs, labels):
        # Plotting decision regions
        X = self.__train_x
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

        for idx, clf, tt in zip(product([0, 1], [0, 1]),
                                clfs, labels):

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
            axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                          s=20, edgecolor='k')
            axarr[idx[0], idx[1]].set_title(tt)

        plt.show()
        
    def plot_conf_mat(self, cls, key):
        disp = plot_confusion_matrix(cls, self.__test_x, self.__test_y)
        disp.ax_.set_title(key)

        print(key)
        print(disp.confusion_matrix)

        plt.show()