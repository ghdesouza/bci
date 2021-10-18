import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys

if not sys.warnoptions:
    import warnings
    warnings.filterwarnings("ignore")

seed = 141114
np.random.seed(seed)

from libs.gp.math_gp import math_gp
from libs.gp.math_gp import plot_graph
from libs.bci.see import see
from libs.kfold import KFold

num_kfolds = 5
folds = 5

#name_file = 'data/parsed_P05T.mat'
name_file = sys.argv[1]
origin = loadmat(name_file)
X = np.array(origin['RawEEGData'])
y = np.array(origin['Labels']).reshape(-1)
channels = ['F3', 'FC3', 'C3', 'CP3', 'P3', 'FCz', 'CPz', 'F4', 'FC4', 'C4', 'CP4', 'P4']
X = X[:, :, 1536:3840] # window selection: 3.0-7.5 sec

X_train, y_train = X, y
clf = see(bandpass=[12, 54])

def fitness_func(func):
    clf.set_features_construction(func)
    clf.fit(X_train, y_train)
    return clf.log_loss(X_train, y_train)
    #return -clf.kappa_score(X_train, y_train)

best_expr = ''
s = 0
results = []
for ran_stat_id in range(num_kfolds):
    kf = KFold(balanced=True, random_state=(seed+ran_stat_id))
    for train_index, test_index in kf.split(X, y):
        X_train, X_test,  y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        clf.set_electrode(-1)
        clf.fit(X_train, y_train)
        
        s += 1
        results.append([s, clf.kappa_score(X_test, y_test), clf.score(X_test, y_test), clf.log_loss(X_test, y_test)])
        print('%2d, %.3f, %.3f, %.6f'%(results[-1][0], results[-1][1], results[-1][2], results[-1][3]))

results = pd.DataFrame(results, columns=['fold', 'kappa', 'accuracy', 'log_loss'])
results.to_csv(sys.argv[2]+".csv", index=False)
