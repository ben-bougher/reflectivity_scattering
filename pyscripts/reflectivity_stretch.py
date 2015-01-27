
import numpy as np
from matplotlib import pyplot as plt
import agilegeo as ag
import scipy.io 

"""
A script that generates two simple reflectivity series, one in time
and one in depth
"""

nsamps = 2096
dt = 0.001 # s
dz = None # m 

wavelet = ag.wavelet.ricker(0.1, dt, 20)

# Make a velocity gradient
vmodel = np.linspace(2000.,5000.,nsamps)

# initialize the series
ref_time = np.zeros(nsamps)

# put in a delta
ref_time[200] = 1.0

# put in a ramp
ref_time[500:600] = np.linspace(0,1,100)
#ref_time[600:699] = np.linspace(1,0,100)[1:]

# put in a square root
ref_time[900:1000] = np.sqrt(np.linspace(0,1,100))
#ref_time[1000:1099] = np.sqrt(np.linspace(1,0,100))[1:]

# a square
#ref_time[1500:1700] = 1.0

seis_time = np.convolve(wavelet, ref_time, mode='same') / wavelet.size

# convert to depth
ref_depth = ag.avo.time_to_depth(ref_time, vmodel, dt)
seis_depth = ag.avo.time_to_depth(seis_time, vmodel, dt)

# zero pad the ime reflectivity to be the same length
#ref_time = np.pad(ref_time, (0,ref_depth.size - ref_time.size),
#                  "constant")

struct = {"ref_time": ref_time,
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



