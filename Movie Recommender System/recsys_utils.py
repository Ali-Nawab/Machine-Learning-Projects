import numpy as np
import pandas as pd
from numpy import loadtxt
import urllib.request

def normalizeRatings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).
    Only include real ratings R(i,j)=1.
    [Ynorm, Ymean] = normalizeRatings(Y, R) normalized Y so that each movie
    has a rating of 0 on average. Unrated moves then have a mean rating (0)
    Returns the mean rating in Ymean.
    """
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return(Ynorm, Ymean)

def load_precalc_params_small():

    url = 'https://raw.githubusercontent.com/kaieye/2022-Machine-Learning-Specialization/main/Unsupervised%20learning%20recommenders%20reinforcement%20learning/week2/Practice%20Lab%201/data/small_movies_X.csv'
    X = loadtxt(urllib.request.urlopen(url), delimiter = ",")

    url = 'https://raw.githubusercontent.com/kaieye/2022-Machine-Learning-Specialization/main/Unsupervised%20learning%20recommenders%20reinforcement%20learning/week2/Practice%20Lab%201/data/small_movies_W.csv'
    W = loadtxt(urllib.request.urlopen(url), delimiter = ",")

    url = 'https://raw.githubusercontent.com/kaieye/2022-Machine-Learning-Specialization/main/Unsupervised%20learning%20recommenders%20reinforcement%20learning/week2/Practice%20Lab%201/data/small_movies_b.csv'
    b = loadtxt(urllib.request.urlopen(url), delimiter = ",")
    b = b.reshape(1,-1)
    num_movies, num_features = X.shape
    num_users,_ = W.shape
    return(X, W, b, num_movies, num_features, num_users)
    
def load_ratings_small():
    url = 'https://raw.githubusercontent.com/kaieye/2022-Machine-Learning-Specialization/main/Unsupervised%20learning%20recommenders%20reinforcement%20learning/week2/Practice%20Lab%201/data/small_movies_Y.csv'
    Y = loadtxt(urllib.request.urlopen(url), delimiter = ",")

    url = 'https://raw.githubusercontent.com/kaieye/2022-Machine-Learning-Specialization/main/Unsupervised%20learning%20recommenders%20reinforcement%20learning/week2/Practice%20Lab%201/data/small_movies_R.csv'
    R = loadtxt(urllib.request.urlopen(url), delimiter = ",")
    return(Y,R)

def load_Movie_List_pd():
    """ returns df with and index of movies in the order they are in in the Y matrix """
    url = 'https://raw.githubusercontent.com/kaieye/2022-Machine-Learning-Specialization/main/Unsupervised%20learning%20recommenders%20reinforcement%20learning/week2/Practice%20Lab%201/data/small_movie_list.csv'
    df = pd.read_csv(urllib.request.urlopen(url), header=0, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return(mlist, df)
