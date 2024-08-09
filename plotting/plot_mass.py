#!/usr/bin/python

# by Elizabeth J. Jeffery, updated Aug 8, 2024

## plot histogram of sampleMass output file.
## -> will only plot the first star. Made for 
##    single star project.
## -> Requires .res and .massSamples file in same
##    directory
## -> Will create a histogram of the mass of the 
##    star for stage 3 (post-burnin) called "mystar.png"

# usage on the command line:
#          $ ./plot_mass.py mystar.res

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

filename = sys.argv[1]
rootname = os.path.splitext(filename)[0]
massFile = rootname+'.massSamples'
histfile = rootname+'_hist.png'

# load the files
# res file to get stage information; only plot stage = 3
data    = np.loadtxt(filename,skiprows=1,usecols=[5])
stage   = data
use     = np.where(stage == 3)
# read massSamples file
massDat = np.loadtxt(massFile,skiprows=1,usecols=[0])
mass    = massDat[use]

# calculate stats
avg_mass = np.average(mass)
std_mass = np.std(mass)
mass1    = avg_mass - 6*std_mass
mass2    = avg_mass + 6*std_mass

#print(avg_mass,std_mass)

# histogram plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
ymass, xmass,_ = ax.hist(mass, bins=20, fill=False, histtype='step')

# fill histograms
ax.set_title(rootname)
ax.hist(mass, bins=20, color='black', fill=False, histtype='step')

# set limits and labels
ax.set_xlim([mass1,mass2])
ax.set(xlabel='Mass (Msun)')

# placement and label calculations
mass_tx_x = avg_mass + 1.5*std_mass ; mass_tx_y = 0.90*ymass.max()
a = '%.3f' % avg_mass ; s = '%.4f' % std_mass ; label = a+' +/- '+s
ax.text(mass_tx_x, mass_tx_y, label)

plt.savefig(histfile)
#plt.close('all')
