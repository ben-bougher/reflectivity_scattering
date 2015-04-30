import numpy as np
import pywt
import scipy.io

import train
def synthetic_fractal(alpha, j0, J):
    """
    Makes synthetic signals from using fractal geometries
    """


    coeff = []

    s = np.random.randn(2.**j0+2) * (2**(-j0*alpha))
    coeff.append(s)

    for j in range(j0,J):
        
        coeff.append(np.random.randn(2.**j + 2) * (2.**(-j*alpha)))



    signal = pywt.waverec(coeff,  'db2')
    
    return signal / np.std(signal)



  

#if __name__ == "__main__":

    ## output_file = '../data/fractalsignals.mat'
    ## fractals = {'layer1': .25,
    ##             'layer2': .5,
    ##             'layer3': 0.75,
    ##             'layer4': 1.0}

    ## depth = 10

    ## output = {}
    
    ## for key, value in fractals.items():
    ##     data = []
    ##     for i in range(100):

    ##         data.append(synthetic_fractal(value, 0, depth))


    ##     output[key] = np.transpose(np.array(data)
    
    ## scipy.io.savemat(output_file, {'data': output})



        
    
    

    
