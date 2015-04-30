
import numpy as np
from matplotlib import pyplot as plt
import agilegeo as ag
import scipy.io

import scipy.signal as sig

"""
A script that generates two simple reflectivity series, one in time
and one in depth
"""

duration = 85 #s
dt = 0.001 # s
dz = None # m 
nsamps = duration / dt # n

# Make a 20 Hz Ricker
wavelet = ag.wavelet.ricker(0.1, dt, 20)

# Make a velocity gradient
vmodel = np.linspace(3000.,8000.,nsamps)


# initialize the series
ref_time = np.zeros(nsamps)


# start with random noise

index = [int(5/dt), int(15/dt)]
ref_time[index[0]:index[1]] = np.random.randn(index[1]-index[0])
#ref_time[index[0]:index[1]+1] /= np.var(ref_time[index[0]:index[1]])


# put in a delta series in the first 2000 points
delta = (np.random.randn(200))
index = [int(25/dt), int(35/dt)]
ref_time[np.random.random_integers(index[0],index[1], 200)] = delta
#ref_time[index[0]:index[1]+1] /= np.var(ref_time[index[0]:index[1]])

plt.figure()


# put in a ramp
r_size = 100
ramp = np.linspace(0,1,r_size)

plt.figure()
plt.plot(ramp)


#ramp /= np.var(ramp)
index = [int(45/dt), int(55/dt)]
ramp_ind = np.random.random_integers(index[0],index[1], 200)
for i in ramp_ind:
    ref_time[i : i + r_size] = ramp * np.random.randn()
#ref_time[index[0]:index[1]+1] /= np.var(ref_time[index[0]:index[1]])

# put in a square root
sqrt= np.sqrt(np.linspace(0,1,r_size))
#sqrt = sqrt / np.var(sqrt)
index = [int(65/dt), int(75/dt)]
sqrt_ind = np.random.random_integers(index[0],index[1], 200)
for i in sqrt_ind:
    ref_time[i : i + r_size] = sqrt * np.random.randn()

#ref_time[index[0]:index[1]+1] /= np.var(ref_time[index[0]:index[1]])


# Put in a gaussian reflectors
## index = [int(85/dt), int(95/dt)]
## ramp_ind = np.random.random_integers(index[0],index[1], 200)
## for i in ramp_ind:
##     ref_time[i : i + r_size] =  sig.gaussian(r_size, 10)* np.random.randn()
 
## ref_time[index[0]:index[1]] /= np.var(ref_time[index[0]:index[1]])



seis_time = np.convolve(wavelet, ref_time, mode='same') / wavelet.size

# convert to depth
ref_depth = ag.avo.time_to_depth(ref_time, vmodel, dt)
seis_depth = ag.avo.time_to_depth(seis_time, vmodel, dt)


struct = {"ref_time": ref_time,
          "dt":dt, "nsamps":nsamps,
          "ref_depth": ref_depth,
          "seis_time": seis_time,
          "seis_depth": seis_depth
          }

plt.subplot('221')
plt.plot(ref_time)
plt.subplot('222')
plt.plot(ref_depth)
plt.subplot('223')
plt.plot(seis_time)
plt.subplot('224')
plt.plot(seis_depth)
# write out a matlab file
scipy.io.savemat("../data/reflectivity.mat", struct)

plt.show()



