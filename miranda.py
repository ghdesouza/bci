import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import skew 
from scipy.signal import periodogram
from sklearn.naive_bayes import GaussianNB
import pywt
import sys

#time python3 miranda.py data/parsed_P01T.mat results/miranda/P01_miranda

if not sys.warnoptions:
    import warnings
    warnings.filterwarnings("ignore")

np.random.seed(141114)

from libs.gp.feature_extractor_gp import fe_gp
from libs.bci.miranda_bci import bci
from libs.kfold import KFold

def filterbank(X):
    
    wavelet = pywt.Wavelet('db1')
    Z = []
    for a in range(len(X)):
        x = X[a]
        coeffs = pywt.wavedec(x, wavelet, level=5)
        for b in range(len(coeffs)-1):
            Z.append([coeffs[b][i] if i < len(coeffs[b]) else 0 for i in range(len(x))])
    return np.array(Z)

def initial_features_func(X, fs=512):
    Z = []
    for a in range(len(X)):
        Z.append([ np.mean(X[a][i]) for i in range(len(X[a]))]) ## Signal Amplitude Average || Teoricamente, tem que ser zero 
        Z[-1].extend([ np.std(X[a][i]) for i in range(len(X[a]))]) ## Signal Amplitude Standard Deviation (SD) || Pode fazer sentido usar
        Z[-1].extend([ skew(X[a][i]) for i in range(len(X[a]))]) ## Symmetry || Teoricamente, tem que ser zero
        Z[-1].extend([ np.mean(periodogram(X[a][i], fs=fs, scaling='density')) for i in range(len(X[a]))]) ## Power Spectral Density (PSD) || ?
        Z[-1].extend([ np.sum(np.absolute(X[a][i])) for i in range(len(X[a]))]) ## Signal Curve Length || bem similar a nossa extração
    return Z

num_kfolds = 5
folds = 5

num_inputs=5*5*12 # 5 levels in wavelet and 5 initial features
args_fitness=[]
pop_size=300
num_gen=100
cxpb=0.85
mutpb=0.15
max_depth=6
verbose=False
int_const=None #[0, 10]
float_const=None #[0, 1]
start_depth=[5, 6]#=[1, 3]
subtree_depth=[1, 2]

#name_file = 'data/parsed_P01T.mat'
name_file = sys.argv[1]
origin = loadmat(name_file)
X = np.array(origin['RawEEGData'])
y = np.array(origin['Labels']).reshape(-1)
channels = ['F3', 'FC3', 'C3', 'CP3', 'P3', 'FCz', 'CPz', 'F4', 'FC4', 'C4', 'CP4', 'P4']
X = list(X[:, :, 1536:3840]) # window selection: 3.0-7.5 sec

for a in range(len(X)):
    X[a] = filterbank(X[a])
X = np.array(X)

X_train, y_train = X, y
clf = bci(initial_features_func=initial_features_func, clf=GaussianNB())

def fitness_func(func):
    clf.set_features_construction(func)
    clf.fit(X_train, y_train)
    #return clf.log_loss(X_train, y_train)
    return -clf.auc(X_train, y_train)

best_expr = ''
s = 0
results = []
for ran_stat_id in range(num_kfolds):
    kf = KFold(balanced=True, random_state=(ran_stat_id+1))
    for train_index, test_index in kf.split(X, y):
        X_train, X_test,  y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        logbook, best_expr, best_fitness, func = fe_gp(fitness_func, num_inputs=num_inputs, args_fitness=args_fitness, pop_size=pop_size, num_gen=num_gen, cxpb=cxpb, mutpb=mutpb, max_depth=max_depth, verbose=verbose, int_const=int_const, float_const=float_const, start_depth=start_depth, subtree_depth=start_depth)
        
        clf.set_features_construction(func)
        clf.fit(X_train, y_train)
        
        s += 1
        logbook = pd.DataFrame(logbook)
        logbook.to_csv(sys.argv[2]+'_logbook_%02d.csv'%s, index=False)
        results.append([s, clf.kappa_score(X_test, y_test), clf.score(X_test, y_test), clf.log_loss(X_test, y_test), best_expr])
        print('%2d, %.3f, %.3f, %.6f, %s'%(results[-1][0], results[-1][1], results[-1][2], results[-1][3], best_expr))

results = pd.DataFrame(results, columns=['fold', 'kappa', 'accuracy', 'log_loss', 'best_expr'])
results.to_csv(sys.argv[2]+".csv", index=False)
    




















