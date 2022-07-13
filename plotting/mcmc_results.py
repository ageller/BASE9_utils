#!/usr/bin/python3.8

### output mcmc results to the screen

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

filename = sys.argv[1] 
rootname = os.path.splitext(filename)[0]
#plotfile = rootname+'.png'
#histfile = rootname+'_hist.png'

# load the data into arrays
data    = np.loadtxt(filename,skiprows=1,usecols=[0,1,2,3,5])
age_all = data[:,0]
feh_all = data[:,1]
dis_all = data[:,2]
av_all  = data[:,3]
stage   = data[:,4]

# only read from main run (don't plot burn in)
ind    = np.where(stage == 3)
age    = age_all[ind]
feh    = feh_all[ind]
dis    = dis_all[ind]
av     = av_all[ind]
i      = list(range(0,len(age),1)) # iteration number

# calculate stats
avg_age = np.average(age)     ; std_age = np.std(age)
avg_age_l = np.average(10**age)/1e9     ; std_age_l = np.std(10**age)/1e9
avg_feh = np.average(feh)     ; std_feh = np.std(feh)
avg_dis = np.average(dis)     ; std_dis = np.std(dis)
avg_av  = np.average(av)      ; std_av  = np.std(av)

a = '%.3f' % avg_age ; s = '%.4f' % std_age ; age_lab = a+' +/- '+s
a = '%.3f' % avg_age_l ; s = '%.3f' % std_age_l ; age_lab_l = a+' +/- '+s
a = '%.3f' % avg_feh ; s = '%.4f' % std_feh ; feh_lab = a+' +/- '+s
a = '%.3f' % avg_dis ; s = '%.4f' % std_dis ; dis_lab = a+' +/- '+s
#avg_av1 = avg_av * 1000 ; std_av1 = std_av * 1000
#a = '%.2f' % avg_av1  ; s = '%.2f' % std_av1  ; av_lab  = a+' +/- '+s
a = '%.4f' % avg_av  ; s = '%.4f' % std_av  ; av_lab  = a+' +/- '+s

# display results to screen
print("log(Age):  "+age_lab)
print("Age (Gyr): "+age_lab_l)
print("[Fe/H]:    "+feh_lab)
print("dist:      "+dis_lab)
print("Av:        "+av_lab)
