import pandas as pd
import numpy as np
from sklearn import utils

class KFold:
    def __init__(self, n_splits=5, random_state=None, shuffle=True, balanced=False):
        self.random_state = random_state
        self.n_splits = n_splits
        self.balanced = balanced
        
    def get_n_splits(self):
        return self.get_n_splits
    
    def split(self, X, y=[], groups=None):

        if len(y) != len(X) or self.balanced == False:
            yb = np.array([1 for i in range(len(X))])
        else:
            yb = y
        
        ytemp = np.array(yb)
        yunique = np.unique(ytemp)
        ydf = pd.DataFrame(np.array([['%d'%i for i in range(len(yb))], yb]).T, columns=['id', 'label'])
        ydf = utils.shuffle(ydf, random_state=self.random_state)
        
        ids = []
        for label_iter in yunique:
            ids.append(np.array(ydf[ydf['label']=='%d'%label_iter]['id']).astype(int))
        
        pos_add = 0
        folds = [[] for i in range(self.n_splits)]
        for label_iter in range(len(yunique)):
            for fold_inter in range(len(ids[label_iter])):
                folds[pos_add%len(folds)].append(ids[label_iter][fold_inter])
                pos_add += 1
        
        exit = [[[], []] for i in range(self.n_splits)]
        for exit_iter in range(len(exit)):
            exit[exit_iter][1] = folds[exit_iter]
            exit[exit_iter][1] = utils.shuffle(exit[exit_iter][1], random_state=self.random_state)
            
            for exit_iter_2 in range(len(exit)):
                if exit_iter != exit_iter_2:
                    exit[exit_iter][0].extend(folds[exit_iter_2])
            exit[exit_iter][0] = utils.shuffle(exit[exit_iter][0], random_state=self.random_state)
        
        return exit
