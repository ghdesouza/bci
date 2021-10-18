import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys

if not sys.warnoptions:
    import warnings
    warnings.filterwarnings("ignore")

np.random.seed(141114)

from libs.gp.math_gp import math_gp
from libs.gp.math_gp import plot_graph
from libs.bci.bci import bci
from libs.kfold import KFold

num_kfolds = 5
folds = 5

num_inputs=12
args_fitness=[]
pop_size=300
num_gen=100
cxpb=0.85
mutpb=0.15
max_depth=6
verbose=True
int_const=None #[0, 10]
float_const=None #[0, 1]
start_depth=[1, 3]
subtree_depth=[1, 2]

#name_file = 'data/parsed_P01T.mat'
name_file = sys.argv[1]
origin = loadmat(name_file)
X = np.array(origin['RawEEGData'])
y = np.array(origin['Labels']).reshape(-1)
channels = ['F3', 'FC3', 'C3', 'CP3', 'P3', 'FCz', 'CPz', 'F4', 'FC4', 'C4', 'CP4', 'P4']
X = X[:, :, 1536:3840] # window selection: 3.0-7.5 sec

X_train, y_train = X, y
clf = bci()

def fitness_func(func):
    clf.set_features_construction(func)
    clf.fit(X_train, y_train)
    return clf.log_loss(X_train, y_train)
    #return -clf.kappa_score(X_train, y_train)

best_expr = ''
s = 0
results = []
for ran_stat_id in range(num_kfolds):
    kf = KFold(balanced=True, random_state=(ran_stat_id+1))
    for train_index, test_index in kf.split(X, y):
        X_train, X_test,  y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        logbook, best_expr, best_fitness, func = math_gp(fitness_func, num_inputs=num_inputs, args_fitness=args_fitness, pop_size=pop_size, num_gen=num_gen, cxpb=cxpb, mutpb=mutpb, max_depth=max_depth, verbose=verbose, int_const=int_const, float_const=float_const, start_depth=start_depth, subtree_depth=start_depth)
        
        clf.set_features_construction(func)
        clf.fit(X_train, y_train)
        
        s += 1
        logbook = pd.DataFrame(logbook)
        logbook.to_csv(sys.argv[2]+'_logbook_%02d.csv'%s, index=False)
        results.append([s, clf.kappa_score(X_test, y_test), clf.score(X_test, y_test), clf.log_loss(X_test, y_test)])
        print('%2d, %.3f, %.3f, %.6f'%(results[-1][0], results[-1][1], results[-1][2], results[-1][3]))

results = pd.DataFrame(results, columns=['fold', 'kappa', 'accuracy', 'log_loss'])
results.to_csv(sys.argv[2]+".csv", index=False)
    





















