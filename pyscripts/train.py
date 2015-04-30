#! /usr/bin/env python

"""
Script for reading in scattering coefficients and training in scikitlearn
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.io import loadmat

import os
import yaml

import sys
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

def knn_train(Xtrain, Xtest, ytrain, ytest,
              n_neighbors=5):
    """
    Uses a simple knn training algorithm to classify and plot
    results
    """

    # Make and train the classifier
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(Xtrain, ytrain)

    yhat = knn.predict(Xtest)

    return knn.score(Xtest, ytest), yhat
    

def seperate(X,y, split=0.5):
    """
    Seperates the data into a testing and training set
    """

    # Get the unique enries
    labels = set(y)

    # initialize outputs
    Xtest = []
    Xtrain = []

    ytest = []
    ytrain = []
    
    for label in labels:

        subset = (y==label)
        Xsub = X[subset,:]
        ysub = y[subset]

        ntrain = int(ysub.size*split)
        ntest = ysub.size - ntrain

        if(ntrain >= 1):
            
            Xtrain += list(Xsub[:ntrain,:])
            ytrain += list(ysub[:ntrain])

        if(ntest >= 1):
            
            Xtest += list(Xsub[ntrain:,:])
            ytest += list(ysub[ntrain:])

    return np.array(Xtest), np.array(ytest), np.array(Xtrain), np.array(ytrain)


    
def main(scatfile, params):

    scat = loadmat(scatfile)
    
    X = np.nan_to_num(scat["X"])
    y = np.ravel(scat["y"])

    # convert from weird nested matlab struct
    if np.ndim(y[0]) ==0:
        y = np.array([str(i) for i in y])

    else:
        y = np.array([str(i[0]) for i in y])
    
    # Make testing and training datasets
    Xtest, ytest, Xtrain, ytrain = \
      seperate(X,y,params["preprocess"]["training_split"])

    if "feature_reduce" in params["preprocess"]:
        
        pca = PCA(n_components=params["preprocess"]["feature_reduce"])
        Xtrain = pca.fit_transform(Xtrain)
        Xtest = pca.transform(Xtest)

    classifier = params["classification"]


    knn_score, yhat = knn_train(Xtrain, Xtest, ytrain, ytest,
                          n_neighbors=classifier["knn"]["k"])

    
    return knn_score, Xtest, ytest, yhat
                            
    
if __name__=='__main__':

    config_file = sys.argv[1]
    filebase = os.path.splitext(os.path.basename(config_file))[0]
    
    sig_file = os.path.join('../data', filebase + '_data.mat')
    scat_file = os.path.join('../data', filebase + '_scat.mat')

    with open(config_file) as f:
    
        # use safe_load instead load
        dataMap = yaml.safe_load(f)

    main(scat_file, dataMap["machine_learning"])


        
    
    

    
    
