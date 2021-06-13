from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from DataSplitter import Splitter
import numpy as np

class ClassifierWrapper:
    def __init__(self, classifier, key):
        self.classifier = classifier
        self.key = key
        self.stats = None
        self.conf_mat = None

class ClassifiersTuner:
    def __init__(self, classifiers, data):
        self.__classifiers = self.assign_classifiers(classifiers)
        
        self.__data = data
        self.__splitter = Splitter(data)
        self.__train_x, self.__train_y, self.__test_x, self.__test_y = self.__splitter.split_train_test()
        
        self.__plotter = Plotter()
        
        self.__picked_classifier = None
        self.__stats = None
        self.__conf_mat = None
        
    def assign_classifiers(self, classifiers):
        clss = {}
        for key in classifiers:
            clss[key] = ClassifierWrapper(classifiers[key], key)
            
        return clss
        
    def run(self, key):
        self.__picked_classifier = self.__classifiers[key].classifier
        self.__stats = None
        self.__conf_mat = None
        
        self.__conf_mat = self.fit_choosen()
        self.__stats = self.calc_stats(key)
        self.draw_conf_mat(key)
        print(self.__stats)
        
        self.__classifiers[key].classifier = self.__picked_classifier
        self.__classifiers[key].conf_mat = self.__conf_mat
        self.__classifiers[key].stats = self.__stats
        
        self.__picked_classifier = None
        self.__stats = None
        self.__conf_mat = None
        
    def plot_full_featured_conf_matrix(self, key):
        clf = self.__classifiers[key].classifier
#         clf.fit(self.__train_x, self.__train_y)
        pred_y = clf.predict(self.__test_x)
        print('self.__test_y: ', self.__test_y)
        print('pred_y: ', pred_y)
        plot_confusion_matrix(clf, self.__test_x, pred_y, normalize='true')
    
    def fit_choosen(self):
#         print('[INFO] Classifier: '+ key)
        self.__picked_classifier.fit(self.__train_x, self.__train_y)
        pred_y = self.__picked_classifier.predict(self.__test_x)
        return confusion_matrix(self.__test_y, pred_y)
    
    def calc_stats(self, key):
        tmp = []
        tmp.append({'cls': key})
        tmp.append({'accuracy': self.accurace(self.__conf_mat)})
        tmp.append({'f1': self.f1(self.__conf_mat)})
        return tmp
    
    def draw_conf_mat(self, key):
        disp = plot_confusion_matrix(self.__picked_classifier, self.__test_x, self.__test_y)
        disp.ax_.set_title('Classifier '+ key)
        plt.show()
        
    def accurace(self, confusion_matrix):
        return (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])
    
    def f1(self, confusion_matrix):
        cls1 = (2 * confusion_matrix[0][0])/(2 * confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[0][1])
        cls2 = (2 * confusion_matrix[1][1])/(2 * confusion_matrix[1][1] + confusion_matrix[1][0] + confusion_matrix[0][1])
        return (cls1 + cls2) / 2
    
    def get(self, key):
        return self.__classifiers[key]
    
    def boundary_plot(self, key):
        self.__plotter.boundary_plot(key, self.get(key).classifier, self.__test_x, self.__test_y) 
    
    
class Plotter():        
    def boundary_plot(self, key, clf, x, y):
        for feature_1 in x.columns:
            for feature_2 in x.columns:
                if feature_1 != feature_2:
                    X = np.array(x[[feature_1,feature_2]].copy())
                    title = str(clf)[:-2] + ' [' + feature_1 + ']' + '[' + feature_2 + ']' + ' boundaries graph'
                    self.plot_decision_boundaries(*self.calc_decision_boundaries(X, y, clf), title)
        
    def make_meshgrid(self, x, y, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy
        
    def plot_contours(self, ax, clf, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out
    
    def plot_decision_boundaries(self, X, y, clf, title="Boundaries Graph"):
        plt.figure(figsize=(20,10))
        fig, ax = plt.subplots()
    #     plt.subplots_adjust(wspace=0.4, hspace=0.4)

        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = self.make_meshgrid(X0, X1)

        self.plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        plt.show()
        
    def calc_decision_boundaries(self, X, y, clf):
        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        clf.fit(X, y)
        return (X, y, clf)