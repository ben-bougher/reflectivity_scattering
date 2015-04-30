import numpy as np
from numpy import abs, convolve, arange

from scipy.signal import firwin as fir, hann

from matplotlib import pyplot as plt


import json

outfile = 'demo.json'

fs = 2048.0
ny = fs/2
ntaps = 128
duration = 1.0 # [s]
t = arange(0,duration, 1/fs)

x = np.random.randn(duration * fs)

# Make the filters
LP = hann(128)

w1 = fir(128, [100.0, 200.0], nyq=ny, pass_zero=False)
w2 = fir(128, [200.0, 500.0], nyq=ny, pass_zero=False)
w3 = fir(128, [500.0, 1000.0], nyq=ny, pass_zero=False)


S0 = convolve(LP,x, mode='same')

x01 = abs(convolve(w1,x, mode='same'))
x02 = abs(convolve(w2,x, mode='same'))
x03 = abs(convolve(w3,x, mode='same'))


x11 = abs(convolve(w1,x01, mode='same'))
x12 = abs(convolve(w2,x01, mode='same'))
x13 = abs(convolve(w3,x01, mode='same'))

x21 = abs(convolve(w1,x02, mode='same'))
x22 = abs(convolve(w2,x02, mode='same'))
x23 = abs(convolve(w3,x02, mode='same'))

x31 = abs(convolve(w1,x03, mode='same'))
x32 = abs(convolve(w2,x03, mode='same'))
x33 = abs(convolve(w3,x03, mode='same'))


js_data = dict(x=tuple(x), LP=tuple(LP), S0=tuple(S0), x01=tuple(x01),
               x02=tuple(x02), x03=tuple(x03), x11=tuple(x11),
               x12=tuple(x12), x13=tuple(x13), x21=tuple(x21),
               x22=tuple(x22), x23=tuple(x23), x31=tuple(x31),
               x32=tuple(x32),x33=tuple(x33),
               t=tuple(t), filtert=tuple(arange(128)),
               w1=tuple(w1),w2=tuple(w2),w3=tuple(w3))

with open(outfile,'w') as f:

    f.write("var data="+json.dumps(js_data))


