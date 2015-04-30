import numpy as np
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec

from synthetics import synthetic_fractal

import pywt

from scipy.optimize import curve_fit

from process import plot_results

import scipy.signal as signal

from pylab import *

from scipy import interpolate

"""
brownian() implements one dimensional Brownian motion (i.e. the Wiener process).
"""

# File: brownian.py

from math import sqrt
from scipy.stats import norm
import numpy as np


def brownian(x0, n, dt, delta, H=1.0, out=None):
    """\
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2H * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=(delta**H)*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


def fit(x, A, B):

    return A*x + B




def fractal_analysis(filename):


    # Make the grid spec
    gs = gridspec.GridSpec(6,4)
    fig = plt.figure()
    
    # Make a synthetic brownian signa
    n = 2**12.
    dt = 1/n
    delta = 1.0
    H=.7
    data = brownian(0.0, n, dt, delta, H)

    # Plot the fractal signal
    ax = plt.subplot(gs[2:4,0])
    ax.plot(np.linspace(0,1,data.size), data)
    ax.set_yticks([])
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x(t)$')
    ax.set_xticks([])
    # Do the wavelet decomposition
    W = pywt.wavedec(data, 'db1')

    offset = 0
    W_interp = []
    for offset, scale in enumerate(W[2:]):

        scale/=np.amax(scale)
        x = np.linspace(0,1, scale.size)
        W_interp.append(np.interp(np.linspace(0,1,5000), x, scale))
        

    W_interp = np.flipud(np.array(W_interp))

    ax = plt.subplot(gs[:3,1])
    ax.imshow(np.abs(W_interp), aspect='auto', cmap="Greys",
               extent = [0,1,1,len(W)],
                interpolation="spline16")
    ax.set_yscale("log", basey=2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('$u$')
    ax.set_ylabel('$log_2(\lambda)$')
    

    f = interpolate.interp1d(np.logspace(0,W_interp.shape[0],
                                          W_interp.shape[0],base=2),
                             W_interp, axis=0, fill_value=0)
    
    W_interp = f(np.logspace(0,11,50, base=2))


    # Modulus magnitude
    ax = plt.subplot(gs[3:,1])
    
    # Find the peaks
    for offset, scale in enumerate(np.flipud(W_interp)):

        peakind = signal.find_peaks_cwt(abs(scale), np.arange(1,100))


        d = np.ones(len(peakind)) + offset

        # interpolate to make a pretty plot
        ax.plot(peakind,d+offset, 'b.', ms=.5)
        
    ax.set_yscale("log", basey=2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('$u$')
    ax.set_ylabel('$log_2(\lambda)$')

    
    ax = plt.subplot(gs[:3,2])
    
    # Hack the partition function
    for q in np.linspace(-10,10,5):

        ax.plot(np.arange(100,0, -1)*q, label='q= ' + str(q))

    ax.legend(loc=4, prop={'size':6})

    ax.set_ylabel('$Z(\lambda,q)$')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('$log_2(\lambda)$')
    
        
    ax = plt.subplot(gs[3:,2])
    
    # Plot the scaling exponent
    ax.plot(np.linspace(-10,10,10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('$q$')
    ax.set_ylabel('$\\tau(q)$')
    
    ax = plt.subplot(gs[2:4,3])
    
    # Plot the 'legendre' transform
    x=np.linspace(-10,10,100)
    ax.plot(-x**2)
    ax.set_xlim((-50,150))
    ax.set_yticks([])
    ax.set_xticks([])

    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$D(\\alpha)$')

    fig.tight_layout()
    plt.savefig(filename)
    
def stat_frac_sim(nscales, outfile):


    fig = plt.figure()

    gs = gridspec.GridSpec(nscales, 1)
    
    for i in range(nscales):
        
        data = np.random.randn(512)
        ax = plt.subplot(gs[i])
        ax.plot(np.linspace(0,1.,512),data)

        if i < nscales -1:
            ax.plot([.42,.42], [-5,5], '--', color='red')
            ax.plot([.61,.61], [-5,5], '--', color='red')
            
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim((-5,5))
        ax.set_xlim((0,1))
        ax.set_ylabel('scale [$\lambda_'+ str(i) +'$]')

        # Zoom lines
        if i>0:
            l1 = matplotlib.lines.Line2D([.45, .13], [.73 -((i-1)*.21), 1-(.21*(i) + .15)],
                transform=fig.transFigure, figure=fig, color='red', linewidth=3)

            l2 = matplotlib.lines.Line2D([.6, .9],[.73 -((i-1)*.21), 1-(.21*(i) + .15)],
                transform=fig.transFigure, figure=fig, color='red', linewidth=3)

            fig.lines.extend([l1, l2])
    
            fig.canvas.draw()


    plt.savefig(outfile)


def stat_frac_multi(nscales, alpha, outfile):


    fig = plt.figure()

    gs = gridspec.GridSpec(nscales, 1)
    
    for i in range(nscales):


        data = synthetic_fractal(alpha, i,9)
   
     
             
        ax = plt.subplot(gs[i])
        ax.plot(np.linspace(0,1., data.size),data)

        if i < nscales -1:
            ax.plot([.42,.42], [-5,5], '--', color='red')
            ax.plot([.61,.61], [-5,5], '--', color='red')
            
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim((-5,5))
        ax.set_xlim((0,1))
        ax.set_ylabel('scale [$\lambda_'+ str(i) +'$]')

        # Zoom lines
        if i>0:
            l1 = matplotlib.lines.Line2D([.45, .13], [.73 -((i-1)*.21), 1-(.21*(i) + .15)],
                transform=fig.transFigure, figure=fig, color='red', linewidth=3)

            l2 = matplotlib.lines.Line2D([.6, .9],[.73 -((i-1)*.21), 1-(.21*(i) + .15)],
                transform=fig.transFigure, figure=fig, color='red', linewidth=3)

            fig.lines.extend([l1, l2])
    
            fig.canvas.draw()


            
    plt.savefig(outfile)


def holder_analysis_plot(alpha, j0, J, outfile):

    signal = synthetic_fractal(alpha, j0,J)

    W = pywt.wavedec(signal, 'db1')[3:]
    W.reverse()
    X = [np.var(i) for i in W]

    plt.figure()
    gs = gridspec.GridSpec(2,len(W)+2, height_ratios = [5,1])

    ax = plt.subplot(gs[0,:2])

    t = np.linspace(0,1, signal.size)
    ax.plot(signal, t, color='g')
    ax.invert_yaxis()

    ax.set_ylabel('t [s]')

    ax.set_xticks([])

    ax.set_title("$\lambda_0$")
    for i,w in enumerate(W):

        ax = plt.subplot(gs[0,i+2])
        t = np.linspace(0,1,w.size)
        ax.plot(w,t)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title('$\lambda_{-'+str((i+1))+'}$')
        

    ax = plt.subplot(gs[1,2:])
    holder, = ax.plot(np.log2(X), label = 'Holder coeff')

    A,B = curve_fit(fit, np.arange(len(X)), np.log2(X))[0]

    hurst, = ax.plot(A*np.arange(len(X)) + B, 'r--', label='Hurst parameter')

    ax.set_yticks([])

    ax.set_yticks([], minor=True)
    ax.set_xticklabels(['$\lambda_{-'+str((i+1))+'}$' for i, dummy in enumerate(W)])


    ax.set_ylabel('$log(\sigma_\lambda)$')


    
    plt.savefig(outfile)

def scatter_plot(outfile):

    # Make a large 3 column gridspec
    fig = plt.figure()

    gs = gridspec.GridSpec(24, 3)

    signal = synthetic_fractal(2.0, 1, 10)

    
    ax = plt.subplot(gs[8:12,0])
    ax.plot(signal)
    ax.set_xlim(0, signal.size)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title('$x(t)$')
    ax.set_xlabel('$t$')

        
    S0 = np.mean(signal)
    ax = plt.subplot(gs[22:,0])
    ax.set_title('$S_0$')
    

    ax.stem([S0], linefmt='b-', markerfmt='bo', basefmt='r-')

    ax.set_xlim((-1,1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('$\mu(x(t))$')
    

    lambda1 = np.abs(pywt.wavedec(signal, 'db1'))


    S1 = [np.mean(i) for i in lambda1[-3:]]
    ax = plt.subplot(gs[22:,1])
    markerline, stemlines, baseline = ax.stem(S1, '-.', bottom=0)
    ax.set_xlim((-1,3))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('$\lambda_i$')
    ax.set_ylabel('$\mu(|\lambda_i|)$')
    ax.set_title('$S_1$')
    

    S2 = []

    for i in np.arange(1,4):

        ax = plt.subplot(gs[8*(i-1):8*(i-1)+2, 1])
        ax.plot(lambda1[-i])
        ax.set_xlim(0, lambda1[-i].size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('$t$')

        ax.set_title('$|\lambda_'+str(i) + '|$')

        
        lambda2 = np.abs(pywt.wavedec(lambda1[-i], 'db1'))

        S2 += [np.mean(k) for k in lambda2[-3:]]

        
        for j in np.arange(1,4):
            ax = plt.subplot(gs[(8*(i-1)) + j-1, 2])
            ax.plot(lambda2[-j])
            ax.set_xlim(0, lambda2[-j].size)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_ylabel('$|\lambda_{'+str(i)+str(j)+'}|$', rotation=0,
                          fontsize='14', labelpad=20)
            ax.yaxis.set_label_position("right")

            if j==3:
                ax.set_xlabel('$t$')

        
    ax = plt.subplot(gs[22:,2])
    markerline, stemlines, baseline = ax.stem(S2, '-.', bottom=0)
    ax.set_xlim((-1,9))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('$S_2$')
    ax.set_xlabel('$\lambda_{ij}$')
    ax.set_ylabel('$\mu(|\lambda_{ij}|)$')


    # Scatter level lines
    l4 = matplotlib.lines.Line2D([.355,.355], [0,1],
                    transform=fig.transFigure, figure=fig, color='green',
                    linestyle='--',
                    linewidth=3)

    l5 = matplotlib.lines.Line2D([.630,.630], [0,1],
                    transform=fig.transFigure, figure=fig, color='green',
                    linestyle='--',
                    linewidth=3)
    
    l1 = matplotlib.lines.Line2D([.355,.395], [.58,.6],
                    transform=fig.transFigure, figure=fig, color='black',
                    linewidth=1)


    l2 = matplotlib.lines.Line2D([.355,.395], [.58,.85],
                    transform=fig.transFigure, figure=fig, color='black',
                    linewidth=1)

    l3 = matplotlib.lines.Line2D([.355,.395], [.58,.35],
                    transform=fig.transFigure, figure=fig, color='black',
                    linewidth=1)


    l6 = matplotlib.lines.Line2D([.630,.67], [.6,.62],
                    transform=fig.transFigure, figure=fig, color='black',
                    linewidth=1)


    l7 = matplotlib.lines.Line2D([.630,.67], [.6,.58],
                    transform=fig.transFigure, figure=fig, color='black',
                    linewidth=1)

    l8 = matplotlib.lines.Line2D([.630,.67], [.6,.56],
                    transform=fig.transFigure, figure=fig, color='black',
                    linewidth=1)

    l9 = matplotlib.lines.Line2D([.630,.67], [.86,.88],
                            transform=fig.transFigure, figure=fig, color='black',
                        linewidth=1)


    l10 = matplotlib.lines.Line2D([.630,.67], [.86,.85],
                    transform=fig.transFigure, figure=fig, color='black',
                    linewidth=1)

    l11 = matplotlib.lines.Line2D([.630,.67], [.86,.83],
                    transform=fig.transFigure, figure=fig, color='black',
                    linewidth=1)

    l12 = matplotlib.lines.Line2D([.630,.67], [.32,.32],
                            transform=fig.transFigure, figure=fig, color='black',
                        linewidth=1)


    l13 = matplotlib.lines.Line2D([.630,.67], [.32,.35],
                    transform=fig.transFigure, figure=fig, color='black',
                    linewidth=1)

    l14 = matplotlib.lines.Line2D([.630,.67], [.32,.29],
                    transform=fig.transFigure, figure=fig, color='black',
                    linewidth=1)


    l15 = matplotlib.lines.Line2D([0,1], [0.21,.21],
                    transform=fig.transFigure, figure=fig, color='red',
                    linestyle='--',
                    linewidth=3)

    
    fig.lines.extend([l1,l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13,
                      l14, l15])

    # End line
    fig.canvas.draw()

    # Coefficient plots
    plt.savefig(outfile)


def knn_example(outfile):


    fig = plt.figure()
    
    gs  = gridspec.GridSpec(6,3, width_ratios=[1,5,1])

    alpha = 2
    J = 9
    j0=7


    trained = [np.random.randn(32) for i in range(12)]

    test = np.random.randn(32)

    ax1 = plt.subplot(gs[:5,0])
    ax1.set_title('Input')
    #ax1.set_ylabel('Feature Coefficients')
    ax1.hlines(np.arange(test.size),0, test, color='black')
    ax1.plot(test, np.arange(test.size), 'o', color='green')
    ax1.plot(np.zeros(test.size), np.arange(test.size), color='blue')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylim((-1,test.size))
    ax1.set_ylabel('Feature Coefficients')

    ax = plt.subplot(gs[:5,1])
    for i, train in enumerate(trained):

        train += i*5
        ax.hlines(np.arange(train.size),i*5, train, color='black')

        
        if i>0 and i % 3 == 0:
            color='yellow'
        else:
            color='red'
            
        ax.plot(train, np.arange(train.size), 'o', color=color)

            
        ax.plot(np.ones(train.size)*i*5, np.arange(train.size), color='blue')
        

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim((-3, 12*5 ))
    ax.set_ylim((-1,32))


    ax.set_xlabel('Samples')
    ax.set_title('Training Data')

    # make the classes plot
    ax = plt.subplot(gs[5,1])

    n = np.arange(12)

    ax.plot(np.array([3,6,9])*5, [1,1,1], 'ys', ms=12)
    ax.plot(np.array([0,2,5,10])*5, [1,1,1,1], 'bD', ms=12)
    ax.plot(np.array([1,4])*5, [1,1], '^', color='purple', ms=12)
    ax.plot(np.array([7,8,11])*5, [1,1,1], 'o', color='black', ms=12)
    

    
    ax.set_xlim((-3,12*5))
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel('Classification Labels')

    ax = plt.subplot(gs[5,0])

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim((0,4))
    ax.set_ylim((0,6))

    ax.text(2,3, '?', fontsize=30, horizontalalignment='center',
            verticalalignment='center')

    ax.plot(1,2, 'ys', ms=9)
    ax.plot(1,4, 'bD', ms=9)
    ax.plot(3,2, '^', color='purple', ms=9)
    ax.plot(3,4, 'o', color='black', ms=9)
    ax.set_xlabel('Classification')


    # Output
    ax = plt.subplot(gs[2,2])

    
    ax.plot(1,1, 'ys', ms=30)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title('Output')

    #ax.set_ylim([0,2])
    #ax.set_xlim([0,2])
    #ax.axis('off')
    
    fig.tight_layout()

    
    fig.savefig(outfile)
    

    
def confusion_good(outfile):

    y = ['A']*20 + ['B']*12 + ['C'] *60 + ['D']*40

    yhat = y

    score = 1.0

    fig = plot_results(np.array(y),np.array(yhat), score)

    fig.savefig(outfile)

     

def confusion_bad(outfile):

    y = ['A']*20 + ['B']*12 + ['C'] *60 + ['D']*40

    yhat = ['A', 'B', 'C','D'] *5 + ['A','B', 'C', 'D']*3 + \
      ['A','B','C', 'D']*15 + ['A', 'B', 'C', 'D']*10

    score = 0.25

    fig = plot_results(np.array(y),np.array(yhat), score)

    fig.savefig(outfile)


    
    
if __name__ == "__main__":

    ## n_scales = 4
    ## stat_frac_sim(n_scales,'../presentation/selfsimilar_stats.eps')

    alpha = 2.
    ## stat_frac_multi(n_scales,alpha, '../presentation/multifractal_stats.eps')
    
    
    
    #scatter_plot('../presentation/scatter_analysis.eps')
    #holder_analysis_plot(alpha, 6, 11, '../presentation/holder_analysis.eps')

    #knn_example( '../presentation/knn.eps')

    #confusion_good('../presentation/good_confusion.eps')
    #confusion_bad('../presentation/bad_confusion.eps')

    fractal_analysis('../presentation/fractal_test2.eps')
