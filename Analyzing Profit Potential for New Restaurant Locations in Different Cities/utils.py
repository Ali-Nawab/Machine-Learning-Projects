import numpy as np

def load_data():
    data = np.loadtxt("x.txt", delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y

def load_data_multi():
    data = np.loadtxt("y.txt", delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y