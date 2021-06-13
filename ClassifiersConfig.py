from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import random

# classifiers params
lr_params = {
    'solver': 'lbfgs',
    'penalty': 'l2',
    'C': 0.1,
}

svc_params_p = {
    'probability': True,
    'kernel': 'poly',
    'degree': 4,
    'gamma': 'auto',
}

svc_params_c = {
    'probability': False,
    'kernel': 'rbf',
    'C': 3.71999208869868,
    'degree': 1.7277025859909652,
    'coef0': 2.6553342650368896,
    'gamma': 'auto',
}

nusvc_params = {
    'probability': True,
    'kernel': 'poly',
    'degree': 4,
    'gamma': 'auto',
    'coef0': 0.053,
    'nu': 0.59,
}

knn_params = {
    'n_neighbors': 17,
    'p': 2.9,
    'n_jobs': -1
}

# 16, 'distance', 'ball_tree', 56, 2

mlp_params = {
    'activation': 'relu',
    'solver': 'lbfgs',
    'tol': 1e-6,
    'hidden_layer_sizes': (250, ),
}

qda_params = {
    'reg_param': 0.111,
}


rnn_params = {
    'n_estimators': 158,
    'criterion': 'entropy',
    'max_depth': None,
    'min_samples_split': 7,
    'min_samples_leaf': 2,
    'min_weight_fraction_leaf': 0.0,
    'max_features': 'sqrt',
    'n_jobs': -1
}

etc_params = {
    'n_estimators': 158,
    'criterion': 'entropy',
    'max_depth': None,
    'min_samples_split': 7,
    'min_samples_leaf': 2,
    'min_weight_fraction_leaf': 0.0,
    'max_features': 'sqrt',
    'n_jobs': -1
}

params_p = {
    'SVC': svc_params_p,
    'DecisionTreeClassifier': {},
    'QuadraticDiscriminantAnalysis': qda_params,
    'GradientBoostingClassifier': {},
    'NaiveGaussian': {},
    'RandomForestClassifier': rnn_params,
    'KNeighborsClassifier': knn_params,
}

params_c = {
    'SVC': svc_params_c,
    'DecisionTreeClassifier': {},
    'QuadraticDiscriminantAnalysis': qda_params,
    'GradientBoostingClassifier': {},
    'NaiveGaussian': {},
    'RandomForestClassifier': rnn_params,
    'KNeighborsClassifier': knn_params,
}

classifiers_default_c = {
    'SVC': SVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'NaiveGaussian': GaussianNB(),
    'RandomForestClassifier': RandomForestClassifier(n_jobs=-1),
    'KNeighborsClassifier': KNeighborsClassifier(n_jobs=-1),
    'ExtraTreesClassifier': ExtraTreesClassifier(n_jobs=-1)
}

classifiers_default_p = {
    'SVC': SVC(probability=True),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'NaiveGaussian': GaussianNB(),
    'RandomForestClassifier': RandomForestClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(n_jobs=-1)
}

classifiers_tuned_p = {
    'SVC': SVC(**svc_params_p),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(**qda_params),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'NaiveGaussian': GaussianNB(),
    'RandomForestClassifier': RandomForestClassifier(**rnn_params),
    'KNeighborsClassifier': KNeighborsClassifier(**knn_params)
}

classifiers_tuned_c = {
    'SVC': SVC(**svc_params_c),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(**qda_params),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'NaiveGaussian': GaussianNB(),
    'RandomForestClassifier': RandomForestClassifier(**rnn_params),
    'KNeighborsClassifier': KNeighborsClassifier(**knn_params)
}

classifiers_etc = {'ExtraTreesClassifier': ExtraTreesClassifier()}

rnn_params_to_tune = {
    'n_estimators': [100, 200],
    'criterion': ['gini'],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt', 'log2'],
    'n_jobs': [-1]
}

etc_params_to_tune = {
    'n_estimators': [80, 160, 240],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [2, 4, 8],
    'max_features': ['auto', 'sqrt', 'log2'],
    'n_jobs': [-1]
}

ng_params_to_tune = {
    'loss': ['deviance', 'exponential'],
    'n_estimators': [80, 160, 240],
    'criterion': ['friedman_mse', 'mse', 'mae'],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [2, 4, 8],
    'max_features': ['auto', 'sqrt', 'log2']
}

qda_params_to_tune = {
    'reg_param': [0.0, 0.1, 0.2, 0.5]
}

svc_params_to_tune = {
    'probability': [False],
    'kernel': ["linear", "rbf", "poly", "sigmoid"],
    'C': [random.uniform(0.1, 100) for x in range(0, 10, 1)],
    'degree': [random.uniform(1, 5) for x in range(0, 10, 1)],
    'coef0': [random.uniform(1, 5) for x in range(0, 10, 1)],
    'gamma': ['auto'],
}