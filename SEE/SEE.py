import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import sys

from scipy.signal import butter, lfilter

data_file = sys.argv[1]
label_file = sys.argv[2]
asc = sys.argv[3]

def butter_bandpass_filter(data, lowcut, highcut, fs=512, order=5):
    return lfilter(*butter(order, [2*lowcut/fs, 2*highcut/fs], btype='band'), data)

def prod(x):
    return np.log(np.sum(np.absolute(x))/len(x))

def extract(df):
    try:
        lc, hc = float(sys.argv[4]), float(sys.argv[5])
        x = prod(butter_bandpass_filter(np.array(df['EEG']), lc, hc))
    except:
        x = prod(np.array(df['EEG']))
    return x

data = []
x = np.array(pd.read_csv('%s'%(data_file), header=None))
for i in range(len(x)):
    data.append(['%03d'%(i+1), extract(pd.DataFrame(x[i], columns=['EEG']))])
    
data = pd.DataFrame(data, columns=['id', 'energy'])
data['energy'] = data['energy'].astype(float)

data = data.sort_values('energy', axis=0, ascending=(asc=='1'))
data['predict'] = np.array([1+int(i/(len(data)/2)) for i in range(len(data))])
data = data.sort_values('id', axis=0, ascending=True)

try:
    l = np.array(pd.read_csv('%s'%(label_file), header=None)).reshape(-1)
    p = np.array(data['predict'])
    print("%+.4f"%cohen_kappa_score(p,l))
except:
    data.to_csv('%s'%(label_file), index=False, float_format='%.5f', sep='\t')
