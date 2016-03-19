import numpy as np
import csv
#import AttrDict

"""
Load data
"""
def load(fname):
    reader = csv.DictReader(open(fname))
    res = {}
    for row in reader:
        for col, value in row.iteritems():
            res.setdefault(col, []).append(value)
    return res

def load_insample():
    reader = csv.reader(open('in_sample_data.txt'))
    res = []
    for row in reader:
       res.append(row)
    res = np.asarray(res, dtype=float)
    #res = np.genfromtxt('in_sample_data.txt', delimiter='.')
    #print res.shape
    return res

def extract_SO(data):
    return data[:, 1 + 6 * np.arange(100)]
def extract_SH(data):
    return data[:, 2 + 6 * np.arange(100)]
def extract_SL(data):
    return data[:, 3 + 6 * np.arange(100)]
def extract_SC(data):
    return data[:, 4 + 6 * np.arange(100)]
def extract_TVL(data):
    return data[:, 5 + 6 * np.arange(100)]
def extract_IND(data):
    return data[:, 6 + 6 * np.arange(100)]
