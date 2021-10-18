import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from scipy.signal import butter, lfilter
from sklearn.linear_model import LogisticRegression

class bci:
    def __init__(self, channel='auto', bandpass=[0, 0]):
        self.clf = LogisticRegression(C=1e5, solver='lbfgs')
        self.channel = channel
        self.bandpass = bandpass

    def prod(self, x): # feature extractor
        if self.bandpass[1] != 0:
            xtemp = butter_bandpass_filter(x, self.bandpass[0], self.bandpass[1])
        else:
            xtemp = x
        return np.log(np.sum(np.absolute(xtemp))/len(xtemp))
    
    def fit(self, X, y):
        
        if self.channel == 'auto':
            mean = []
            for channel_id in range(len(X[0])):
                self.channel = channel_id
                X_one = X[:, channel_id, :] # separate one electrode
                f = np.array([ [self.prod(X_one[i])] for i in range(len(X_one)) ])
                self.clf.fit(f, y)
                mean.append(cohen_kappa_score(self.predict(X), y))
            mean = np.array(mean)
            self.channel = np.argmax(mean)

        X_one = X[:, self.channel, :] # separate one electrode
        f = np.array([ [self.prod(X_one[i])] for i in range(len(X_one)) ])
        self.clf.fit(f, y)
    
    def predict(self, X):
        X_one = X[:, self.channel, :] # separate one electrode
        f = np.array([ [self.prod(X_one[i])] for i in range(len(X_one)) ])
        return self.clf.predict(f)

    def score(self, X, y):
        return accuracy_score(self.predict(X), y)
    
    def kappa_score(self, X, y):
        return cohen_kappa_score(self.predict(X), y)

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
