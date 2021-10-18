import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from scipy.signal import butter, lfilter
from sklearn.linear_model import LogisticRegression

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs=512, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def basic_preprocessing(X, bandpass=[0, 0]):
    if bandpass[1] != 0:
        return np.array([ butter_bandpass_filter(X[i], bandpass[0], bandpass[1]) for i in range(len(X)) ])
    return np.array(X)

def basic_initial_features(X):
    Z = np.array([np.log(np.mean(np.absolute(X[i]))) for i in range(len(X))])
    return Z

def basic_feature_construction(Z):
    F = np.array(Z)
    return F

class see:
    def __init__(self, electrode_id=-1, preprocessing_func=basic_preprocessing, bandpass=[0, 0], initial_features_func=basic_initial_features, features_construction_func=basic_feature_construction, clf='LogisticRegression', random_state=None):
        
        self.electrode_id = electrode_id
        self.preprocessing_func = preprocessing_func
        self.bandpass = bandpass
        self.initial_features_func = initial_features_func
        self.features_construction_func = features_construction_func
        self.random_state = random_state
        if clf == 'LogisticRegression':
            self.clf = LogisticRegression(C=1e5, solver='lbfgs', random_state=self.random_state)
        else:
            self.clf = clf

    def set_features_construction(self, func):
        self.features_construction_func = func
        
    def set_electrode(self, electrode_id):
        self.electrode_id = electrode_id

    def fit(self, X, y):
                
        # electrode and bandpass choice
        if self.electrode_id < 0 or self.electrode_id >= len(X[0]):
            best_val = -np.inf
            best_id = 0
            best_bandpass = [0, 0]
            for a in range(len(X[0])):
                self.electrode_id = a
                X_train, y_train, X_val, y_val = X[:int(3*len(X)/4)], y[:int(3*len(X)/4)], X[int(3*len(X)/4):], y[int(3*len(X)/4):]
                
                # without bandpass
                Z = X_train[:, self.electrode_id, :]
                self.bandpass = [0, 0]
                Z = self.preprocessing_func(Z)
                Z = self.initial_features_func(Z)
                f = np.array([self.features_construction_func(Z[i]) for i in range(len(Z))])
                f = np.reshape(np.array(f), (len(Z), int(len(f)/len(Z))))
                self.clf.fit(f, y_train)
                validation = self.kappa_score(X_val, y_val)
                if validation > best_val:
                    #print("%d, %d, %d, %.3f, %.3e, %.3f, %.3e"%(a, 0, 0, self.kappa_score(X_train, y_train), self.log_loss(X_train, y_train), validation, self.log_loss(X_val, y_val)))
                    best_val = validation
                    best_id = self.electrode_id
                    best_bandpass = self.bandpass
                
                # with bandpass
                for b in range(4, 60+4, 4):
                    for c in range(b+4, 60+4, 4):
                        Z = X_train[:, self.electrode_id, :]
                        self.bandpass = [b, c]
                        Z = self.preprocessing_func(Z, self.bandpass)
                        Z = self.initial_features_func(Z)
                        f = np.array([self.features_construction_func(Z[i]) for i in range(len(Z))])
                        f = np.reshape(np.array(f), (len(Z), int(len(f)/len(Z))))
                        self.clf.fit(f, y_train)
                        validation = self.kappa_score(X_val, y_val)
                        if validation > best_val:
                            #print("%d, %d, %d, %.3f, %.3e, %.3f, %.3e"%(a, b, c, self.kappa_score(X_train, y_train), self.log_loss(X_train, y_train), validation, self.log_loss(X_val, y_val)))
                            best_val = validation
                            best_id = self.electrode_id
                            best_bandpass = self.bandpass
                
            self.electrode_id = best_id
            self.bandpass = best_bandpass
        
        # bandpass choice
        ## [implemntar aqui]
        
        # final train
        Z = X[:, self.electrode_id, :]
        Z = self.preprocessing_func(Z, self.bandpass)
        Z = self.initial_features_func(Z)
        f = np.array([self.features_construction_func(Z[i]) for i in range(len(Z))])
        f = np.reshape(np.array(f), (len(Z), int(len(f)/len(Z))))
        self.clf.fit(f, y)


    def predict(self, X):
        Z = X[:, self.electrode_id, :]
        Z = self.preprocessing_func(Z, self.bandpass)
        Z = self.initial_features_func(Z)
        f = np.array([self.features_construction_func(Z[i]) for i in range(len(Z))])
        f = np.reshape(np.array(f), (len(Z), int(len(f)/len(Z))))
        return self.clf.predict(f)

    def predict_proba(self, X):
        Z = X[:, self.electrode_id, :]
        Z = self.preprocessing_func(Z, self.bandpass)
        Z = self.initial_features_func(Z)
        f = np.array([self.features_construction_func(Z[i]) for i in range(len(Z))])
        f = np.reshape(np.array(f), (len(Z), int(len(f)/len(Z))))
        return self.clf.predict_proba(f)

    def score(self, X, y):
        return accuracy_score(self.predict(X), y)
    
    def kappa_score(self, X, y):
        return cohen_kappa_score(self.predict(X), y)

    def log_loss(self, X, y):
        return log_loss(y, self.predict_proba(X))
        
