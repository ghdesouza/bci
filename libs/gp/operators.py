import numpy as np
MIN_LOG_X1 = 1
MIN_DIV_VALUE = 1e-8
MAX_POT_EXP = 3
MAX_POT_VALUE = 1e+8

def _add_(x1, x2):
    return x1+x2

def _sub_(x1, x2):
    return x1-x2

def _mul_(x1, x2):
    return x1*x2

def _div_(x1, x2):
    if abs(float(x2)) < MIN_DIV_VALUE:
        return x1
    else:
        return x1/x2

def _neg_(x1):
    return -1*x1

def _F_(x1):
    return x1

def _log_(x1):
    if abs(x1) >= MIN_LOG_X1:
        return np.log(abs(x1))
    return np.log(MIN_LOG_X1)
    
def _sqrt_(x1):
    return abs(x1)**0.5

def _pot_(x1, x2):
        try:
            if x2 > MAX_POT_EXP:
                return min(abs(x1)**MAX_POT_EXP, MAX_POT_VALUE)
            else:
                return min(abs(x1)**x2, MAX_POT_VALUE)
        except:
            return MAX_POT_VALUE

def _mod_(x1):
    return abs(x1)

def _sin_(x1):
    return np.sin(x1)

def _cos_(x1):
    return np.cos(x1)
















