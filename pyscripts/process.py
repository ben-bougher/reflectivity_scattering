#! /usr/bin/env python

import os
import train
import sys
import yaml
import pywt
import numpy as np

from scipy.io import loadmat, savemat
from matplotlib import pyplot as plt

from itertools import product 
"""
Main script for processing testing classifications
"""

def extract_signals(labels, N, output_file):
    """
    Wrapper around matlab file
    """

    os.chdir('../mscripts')
    labstr = '{' + ','.join(["'" + i +"'" for i in labels]) + '}'
    matcom =  'extract_signals(' + ','.join([labstr, str(N), "'" + output_file + "'"]) + ')'
    command = 'matlab -nosplash -nodesktop -r "'+ matcom + ';exit();"' 

    
    os.system(command)


    os.chdir('../pyscripts')

def extract_derivatives(labels, N, output_file):
    """
    Wrapper around matlab file
    """

    os.chdir('../mscripts')
    labstr = '{' + ','.join(["'" + i +"'" for i in labels]) + '}'
    matcom =  'extract_derivative(' + ','.join([labstr, str(N), "'" + output_file + "'"]) + ')'
    command = 'matlab -nosplash -nodesktop -r "'+ matcom + ';exit();"' 

    
    os.system(command)

    os.chdir('../pyscripts') 

def fractal_scatter(infile, outfile, N, M, T):
    """
    Wrapper around matlab files
    """

    matcom =  'fractal_scatter(' + \
      ','.join(["'"+infile+"'", "'"+outfile +"'", str(N),str(T)]) + ')'
    command = 'matlab -nosplash -nodesktop -r "'+ matcom + ';exit();"' 
    os.chdir('../mscripts')
    
    os.system(command)
    os.chdir('../pyscripts')


def scatter_transfer(sigfile, scat_file, N, T):

    matcom =  'scattering_transfer(' + \
      ','.join(["'"+sigfile+"'", "'"+scat_file +"'", str(N), str(T)]) + ')'
    command = 'matlab -nosplash -nodesktop -r "'+ matcom + ';exit();"' 
    os.chdir('../mscripts')
    
    os.system(command)
    os.chdir('../pyscripts')

def holder_exp(sigfile, scatfile):

    d  = loadmat(sigfile)
    data = d["data"]

    X = []
    y = []
    for label in data.dtype.names:

        signals = np.transpose(data[label][0,0])

        for ts in signals:

            W = pywt.wavedec(ts, 'db1')

            holder = X.append([np.var(i) for i in W])
            y.append(label)

    X = np.array(X)
    y = np.array(y)
        
    data = {"X":X, "y":y}

    savemat(scatfile, data)


def plot_results(y,yhat, score):

    classes = sorted(set(y))
    class_dict = dict(zip(classes, np.arange(len(classes))+1))


    xcoord,ycoord = np.mgrid[slice(0,y.size,1), slice(0,y.size,1)]

    confuse = np.zeros(xcoord.shape)

    xind = 0
    yind = 0
    for label1,label2 in product(classes, repeat=2):

        nl1 = sum(y==label1)
        nl2 = sum(y==label2)
        confusion = (sum(np.logical_and(y==label1, yhat==label2))/
                    float((sum(y==label1))))
            

        confuse[xind:xind + nl1, yind:yind + nl2] = \
          np.ones((nl1,nl2))*confusion

        
        yind += nl2
        if yind % confuse.shape[0] ==0:
            yind=0
            xind += nl1
            

    # Make the tick divisions
    ticks = []
    label_pos = []
    total = 0
    for label in classes:
        ticks.append(total + (sum(y==label)))
        label_pos.append(total + (sum(y==label)/2.0))
        total += sum(y==label)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_xticks(ticks)
    ax.set_xticklabels([])
    ax.set_xticks(label_pos, minor=True)
    ax.set_xticklabels(classes, minor=True, rotation=30)
    ax.set_yticks(ticks)
    ax.set_yticks(label_pos, minor=True)
    ax.set_yticklabels([])
    ax.set_yticklabels(classes, minor=True)
    ax.grid(which="major")

    ax.set_xlabel("True Class")
    ax.set_ylabel("Predicted Class")
    ax.set_title('P= ' + '%.2f' % score) 
    plot = ax.imshow(confuse, cmap="afmhot_r", clim=[0,1])
    fig.colorbar(plot)
    fig.tight_layout()

    return fig


def main(config_file):

    with open(config_file) as f:
    
        # use safe_load instead load
        dataMap = yaml.safe_load(f)


    filebase = os.path.splitext(os.path.basename(config_file))[0]
    
    sig_file = os.path.join('../data', filebase + '_data.mat')
    scat_file = os.path.join('../data', filebase + '_scat.mat')

    labels = dataMap["data"]["labels"]
    Nsig = dataMap["data"]["N"]

    if (('derivative' in dataMap["data"]) and
        dataMap['data']["derivative"]):
        extract_derivatives(labels, Nsig, sig_file)
    else:
        extract_signals(labels, Nsig, sig_file)

    if dataMap["features"] == 'holder':
        holder_exp(sig_file, scat_file)
    elif 'scattering_transfer' in  dataMap["features"]:
        
        scat = dataMap["features"]["scattering_transfer"]
        scatter_transfer(sig_file, scat_file, scat["N"], scat["T"])
    else:
        scat = dataMap["features"]["scatter"]
        fractal_scatter(sig_file, scat_file, scat["N"], scat["M"], scat["T"])

    
    score, X, y, yhat = train.main(scat_file, dataMap["machine_learning"])

    # Make the summary plot
    fig = plot_results(y, yhat, score)
    fig.savefig(os.path.join('../results/', filebase +'.eps'))

    dataMap["results"] = str(score)

    outyaml = os.path.join('../results', filebase+'.yaml')

    with open(outyaml, 'w') as f:
        f.write(yaml.dump(dataMap, default_flow_style=False))

    
if __name__=="__main__":


    
    #config_file = '../experiments/exp1.yaml'
    arg = sys.argv[1]

    if arg == "all":

        indir = '../experiments'
        for f in sorted(os.listdir(indir)):

            if not os.path.exists(os.path.join('../results', f)):
                try:
                    main(os.path.join(indir, f))
                except:
                    pass
    else:
        main(arg)


    
  

    
    
    
    

    
