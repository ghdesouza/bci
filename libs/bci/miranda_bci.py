import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def basic_preprocessing(X):
    return X

def basic_initial_features(X):
    Z = []
    for a in range(len(X)):
        Z.append([np.log(np.mean(np.absolute(X[a][i]))) for i in range(len(X[a]))])
    return Z

def basic_feature_construction(F3, FC3, C3, CP3, P3, FCz, CPz, F4, FC4, C4, CP4, P4):
    F = np.array([F3, FC3, C3, CP3, P3, FCz, CPz, F4, FC4, C4, CP4, P4])
    return F

class bci:
    def __init__(self, preprocessing_func=basic_preprocessing, initial_features_func=basic_initial_features, features_construction_func=basic_feature_construction, clf=LogisticRegression(C=1e5, solver='lbfgs')):
        
        self.preprocessing_func = preprocessing_func
        self.initial_features_func = initial_features_func
        self.features_construction_func = features_construction_func
        self.clf = clf

    def set_features_construction(self, func):
        self.features_construction_func = func

    def fit(self, X, y):
        
        Z = self.preprocessing_func(X)
        Z = self.initial_features_func(Z)
        f = []
        for a in range(len(Z)):
            f.append([self.features_construction_func[i](*Z[a]) for i in range(len(self.features_construction_func))])
        f = np.array(f).reshape(-1)
        f = np.reshape(np.array(f), (len(X), int(len(f)/len(X))))
        self.clf.fit(f, y)
    
    def predict(self, X):
        Z = self.preprocessing_func(X)
        Z = self.initial_features_func(Z)
        f = []
        for a in range(len(Z)):
            f.append([self.features_construction_func[i](*Z[a]) for i in range(len(self.features_construction_func))])
        f = np.array(f).reshape(-1)
        f = np.reshape(np.array(f), (len(X), int(len(f)/len(X))))
        return self.clf.predict(f)

    def predict_proba(self, X):
        Z = self.preprocessing_func(X)
        Z = self.initial_features_func(Z)
        f = []
        for a in range(len(Z)):
            f.append([self.features_construction_func[i](*Z[a]) for i in range(len(self.features_construction_func))])
        f = np.array(f).reshape(-1)
        f = np.reshape(np.array(f), (len(X), int(len(f)/len(X))))
        return self.clf.predict_proba(f)

    def score(self, X, y):
        return accuracy_score(self.predict(X), y)
    
    def kappa_score(self, X, y):
        return cohen_kappa_score(self.predict(X), y)

    def log_loss(self, X, y):
        return log_loss(y, self.predict_proba(X))
        
    def auc(self, X, y): # FALTA IMPLEMENTAR AQUI AINDA 
        pred = self.predict_proba(X)[:, 1]
        try:
            return roc_auc_score(y, pred)
        except:
            return 0
        
