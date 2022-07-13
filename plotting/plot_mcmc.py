#!/usr/bin/python3.8

### plot mcmc output of .res files, with distance, rather than parallax

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

filename = sys.argv[1] 
#filename = 'hz4_msrgb4_ifmr1.res'
rootname = os.path.splitext(filename)[0]
plotfile = rootname+'.png'
histfile = rootname+'_hist.png'

# load the data into arrays
data    = np.loadtxt(filename,skiprows=1,usecols=[0,1,2,3,5])
age_all = data[:,0]
feh_all = data[:,1]
plx_all = data[:,2]
av_all  = data[:,3]
stage   = data[:,4]

# only read from main run (don't plot burn in)
ind    = np.where(stage == 3)
age    = age_all[ind]
feh    = feh_all[ind]
plx    = plx_all[ind]
av     = av_all[ind]
i      = list(range(0,len(age),1)) # iteration number

steps  = len(i)/10

# calculate y-limits
avg_age = np.average(age)     ; std_age = np.std(age)
age1    = avg_age - 5*std_age ; age2    = avg_age + 5*std_age
avg_feh = np.average(feh)     ; std_feh = np.std(feh)
feh1    = avg_feh - 5*std_feh ; feh2    = avg_feh + 5*std_feh
avg_plx = np.average(plx)     ; std_plx = np.std(plx)
plx1    = avg_plx - 5*std_plx ; plx2    = avg_plx + 5*std_plx
avg_av  = np.average(av)      ; std_av  = np.std(av)
av1     = avg_av - 3*std_av   ; av2     = avg_av + 5*std_av

# make plot of sampling history
fig, ax=plt.subplots(4,1,sharex=True,figsize=(14,9))
fig.subplots_adjust(hspace=0)
ax[0].set_title(rootname)

# plot values
ax[0].scatter(i,age,marker='o',s=0.1,c='k') #,markersize=1)
ax[1].scatter(i,feh,marker='o',s=0.1,c='k')
ax[2].scatter(i,plx,marker='o',s=0.1,c='k')
ax[3].scatter(i,av,marker='o',s=0.1,c='k')

# label y-axes
ax[0].set(ylabel='Log(Age)')
ax[1].set(ylabel='[Fe/H]')
ax[2].set(ylabel='Distance, (m - M)')
ax[3].set(ylabel='Av')

# set y-limits
ax[0].set_ylim([age1,age2])
ax[1].set_ylim([feh1,feh2])
ax[2].set_ylim([plx1,plx2])
ax[3].set_ylim([av1,av2])

# a few other things to pretty up the plot
plt.xlabel('i')
#plt.xticks(np.arange(min(i),max(i),step=steps))#1000))

# display or save plot
#plt.show()
plt.savefig(plotfile)
plt.close('all')


# make histograms of output
fig, ax=plt.subplots(nrows=2,ncols=2,figsize=(10,10))
ax0,ax1,ax2,ax3 = ax.flatten()

# calculate where to put the text on plot
yage,xage,_ = ax0.hist(age,bins=20,fill=False,histtype='step')
yfeh,xfeh,_ = ax1.hist(feh,bins=20,fill=False,histtype='step')
yplx,xplx,_ = ax2.hist(plx,bins=20,fill=False,histtype='step')
yav,xav,_   = ax3.hist(av*1000,bins=20,fill=False,histtype='step')

# fill histograms
ax0.set_title(rootname)
ax0.hist(age,bins=20,color='black',fill=False,histtype='step')
ax1.hist(feh,bins=20,color='black',fill=False,histtype='step')
ax2.hist(plx,bins=20,color='black',fill=False,histtype='step')
ax3.hist(av*1000,bins=20,color='black',fill=False,histtype='step')
# calculate Gaussian dist. and overplot on age histogram
#dist = np.random.normal(avg_age,std_age,1000)
#ax0.scatter(20,1/(std_age*np.sqrt(2*np.pi)) * np.exp( - (20-avg_age)**2 / (2 * std_age**2) ),linewidth=2,color='r')

# label x-axes
ax0.set(xlabel='Log(Age)')
ax1.set(xlabel='[Fe/H]')
ax2.set(xlabel='m - M')
ax3.set(xlabel='Av (mmag)')

# set x-limits
ax0.set_xlim([age1,age2])
ax1.set_xlim([feh1,feh2])
ax2.set_xlim([plx1,plx2])
ax3.set_xlim([av1*1000,av2*1000])

# placement and labels calculations
age_tx_x = avg_age + 1.4*std_age ; age_tx_y = 0.90*yage.max()
feh_tx_x = avg_feh + 1.4*std_feh ; feh_tx_y = 0.90*yfeh.max()
plx_tx_x = avg_plx + 1.4*std_plx ; plx_tx_y = 0.90*yplx.max()
av_tx_x  = avg_av + 1.4*std_av   ; av_tx_y  = 0.90*yav.max()

a = '%.3f' % avg_age ; s = '%.4f' % std_age ; age_lab = a+' +/- '+s
a = '%.3f' % avg_feh ; s = '%.4f' % std_feh ; feh_lab = a+' +/- '+s
a = '%.3f' % avg_plx ; s = '%.4f' % std_plx ; plx_lab = a+' +/- '+s
avg_av1 = avg_av * 1000 ; std_av1 = std_av * 1000
a = '%.2f' % avg_av1  ; s = '%.2f' % std_av1  ; av_lab  = a+' +/- '+s

# add text of average +/- stdev
ax0.text(age_tx_x,age_tx_y,age_lab)
ax1.text(feh_tx_x,feh_tx_y,feh_lab)
ax2.text(plx_tx_x,plx_tx_y,plx_lab)
ax3.text(av_tx_x*1000,av_tx_y,av_lab)

# display or save histogram plot
#plt.show()
plt.savefig(histfile)
plt.close('all')
