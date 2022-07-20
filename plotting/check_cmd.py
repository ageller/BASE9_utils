#!python

# check the CMD for a BASE-9 phot file

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# read the phot file in
photfile = sys.argv[1]
mag     = sys.argv[2]
color     = sys.argv[3].split('-')

#photfile = "hyades_UBVRIKs.phot"
#mag       = 'V'
#c         = 'B-V'
#color     = c.split('-')

header = np.loadtxt(photfile,max_rows=1,dtype=str)

yplot     = np.where(header == mag)[0][0]
color1    = np.where(header == color[0])[0][0]
color2    = np.where(header == color[1])[0][0]

ylabel = header[int(yplot)]
xlabel = header[int(color1)]+' - '+header[int(color2)]

data = np.loadtxt(photfile,skiprows=1)
nstars = len(data[:,0])
nfilts = int((data.shape[1] - 6)/2)


# include only stars for which phot in all 3 filters exists
good   = np.where((data[:,yplot+nfilts] > 0) & (data[:,color1+nfilts] > 0) & (data[:,color2+nfilts] > 0))

ymag    = data[:,yplot][good]
color   = data[:,color1][good] - data[:,color2][good]
mag_err = data[:,yplot+nfilts][good]
col_err = np.sqrt((data[:,color1+nfilts][good])**2 + (data[:,color2+nfilts][good])**2)

print("\n  Plotting "+ylabel+" vs. "+xlabel+" CMD for "+str(len(ymag))+" / "+str(nstars)+" stars from "+photfile+".\n  File is cmd.png.\n")

N = 0.6
#print(np.max(ymag),np.min(ymag))
if ((np.min(ymag) < 0) & (abs(np.min(ymag)) < 0.1)):
    N = 20
if ((np.min(ymag) < 0) & (abs(np.min(ymag)) > 0.1)):
    N = 2.0
if ((np.min(ymag) > 0) & (abs(np.min(ymag)) < 0.1)):
    N = 0.01
#print(1.1*np.max(ymag),N*np.min(ymag))

plt.figure(figsize=(6,10))
plt.errorbar(color,ymag,xerr=col_err,yerr=mag_err,fmt='o',markersize=2,color='k')
plt.ylim(1.1*np.max(ymag),N*np.min(ymag))
plt.ylabel(ylabel)
plt.xlabel(xlabel)
plt.title(photfile)


if (len(sys.argv) > 4):
    cmdfile = sys.argv[4]
    mag     = sys.argv[2]
    color     = sys.argv[3].split('-')

    header = np.loadtxt(cmdfile,max_rows=1,dtype=str)
    yplot     = np.where(header == mag)[0][0]
    color1    = np.where(header == color[0])[0][0]
    color2    = np.where(header == color[1])[0][0]
    data = np.loadtxt(cmdfile,skiprows=1)
    ymag    = data[:,yplot]
    color   = data[:,color1] - data[:,color2]
    plt.plot(color, ymag, color='red')

plt.savefig('cmd.png')

plt.cla()
plt.clf()
plt.close()
