#!/usr/bin/python3.8

### output results of each sampleWDMass file to the screen

import numpy as np
import sys
import os

filename = sys.argv[1] 
rootname = os.path.splitext(filename)[0]
#plotfile = rootname+'.png'
#histfile = rootname+'_hist.png'

# read the .res file to get the stage info
# only read from main run (don't plot burn in)
stage   = np.loadtxt(filename,skiprows=1,usecols=[5])
use     = np.where(stage == 3)

# read in the individual files
coolAge = np.loadtxt(rootname+'.wd.coolingAge')[use]
logg    = np.loadtxt(rootname+'.wd.logg')[use]
logTeff = np.loadtxt(rootname+'.wd.logTeff')[use]
mass    = np.loadtxt(rootname+'.wd.mass')[use]
memb    = np.loadtxt(rootname+'.wd.membership')[use]
precAge = np.loadtxt(rootname+'.wd.precLogAge')[use]

# calculate stats
avg_cage  = np.average(coolAge)   ; std_cage = np.std(coolAge)
avg_logg  = np.average(logg)      ; std_logg = np.std(logg)
avg_logT  = np.average(logTeff)   ; std_logT = np.std(logTeff)
avg_mass  = np.average(mass)      ; std_mass = np.std(mass)
avg_memb  = np.average(memb)      ; std_memb = np.std(memb)
avg_pAge  = np.average(precAge)   ; std_pAge = np.std(precAge)
avg_cage2 = np.average(10**coolAge/1e9)   ; std_cage2 = np.std(10**coolAge/1e9)
avg_pAge2 = np.average(10**precAge/1e9)   ; std_pAge2 = np.std(10**precAge/1e9)
avg_Teff  = np.average(10**logTeff)   ; std_Teff = np.std(10**logTeff)

a = '%.3f' % avg_cage ; s = '%.4f' % std_cage ; cage_lab = a+' +/- '+s
a = '%.3f' % avg_logg ; s = '%.4f' % std_logg ; logg_lab = a+' +/- '+s
a = '%.3f' % avg_logT ; s = '%.4f' % std_logT ; logT_lab = a+' +/- '+s
a = '%.3f' % avg_mass ; s = '%.4f' % std_mass ; mass_lab = a+' +/- '+s
a = '%.3f' % avg_memb ; s = '%.4f' % std_memb ; memb_lab = a+' +/- '+s
a = '%.3f' % avg_pAge ; s = '%.4f' % std_pAge ; pAge_lab = a+' +/- '+s
a = '%.3f' % avg_cage2; s = '%.4f' % std_cage2 ; cage_lab2 = a+' +/- '+s
a = '%.3f' % avg_pAge2; s = '%.4f' % std_pAge2 ; pAge_lab2 = a+' +/- '+s
a = '%.0f' % avg_Teff ; s = '%.0f' % std_Teff ; teff_lab = a+' +/- '+s

# display results to screen
print("log(Cooling Age):  "+cage_lab+" ("+cage_lab2+" Gyr)")
print("Prec. log(Age):    "+pAge_lab+" ("+pAge_lab2+" Gyr)")
print("log(Teff):         "+logT_lab+" ("+teff_lab+" K)")
print("log(g):            "+logg_lab)
print("Prec. Mass:        "+mass_lab)
print("Membership:        "+memb_lab)
