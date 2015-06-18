import numpy as np

def closeornan(a, b):
    '''Compare two arrays and allow nan==nan'''
    isnan = np.isnan(a)
    isnanb = np.isnan(b)
    if np.all(isnan == isnanb):
        return np.allclose(a[~isnan], b[~isnan])
    else:
        return False
