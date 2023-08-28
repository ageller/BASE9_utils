#!/usr/bin/env python
# coding: utf-8

# In[122]:


import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--name",
    type=str,
    help="Define cluster name after flag -n",
    default="NGC_188"
)

parser.add_argument("-m", "--min_mag",
    type=float,
    help="Define minimum magnitude after flag -m",
    default=0.0
)

args = parser.parse_args()
cluster = args.name
min_mag = args.min_mag
print ('Cluster and min mag are ', cluster, min_mag)
ph= pd.read_csv('../../'+cluster+'.df',sep=' ')
members = ph.loc[(ph['member'] == True)]
length = len(members[(members['G']<=20) & (members['G']>=min_mag)])
filters = ['G','G_BP', 'G_RP', 'g_ps', 'r_ps', 'i_ps', 'z_ps', 'y_ps', 'J_2M', 'H_2M','Ks_2M']

partitions = []
scatters = []
means = []
for i in range(len(filters)):
    scat = []
    me = []
    data = members.loc[(members['sig'+filters[i]]>0) & (members['sig'+filters[i]] < 5)]
    bins = np.arange(min(data[filters[i]]),max(data[filters[i]]),.1)
    partitions.append(bins)
    for j in range(len(bins)-1):
        part = data[(bins[j]<= data[filters[i]]) & (data[filters[i]] < bins[j+1]) ]
        if len(part)==0:
            scat.append(scat[-1])
            me.append(me[-1])
            continue
        scat.append(np.std(part['sig'+filters[i]]))
        me.append(np.mean(part['sig'+filters[i]]))
    scatters.append(scat)
    means.append(me)

cols = ['id', 'G', 'G_BP', 'G_RP', 'g_ps', 'r_ps', 'i_ps', 'z_ps', 'y_ps','J_2M', 'H_2M', 'Ks_2M', 'sigG', 'sigG_BP', 'sigG_RP', 'sigg_ps','sigr_ps', 'sigi_ps', 'sigz_ps', 'sigy_ps', 'sigJ_2M', 'sigH_2M','sigKs_2M', 'mass1', 'massRatio', 'stage1', 'CMprior', 'useDBI']

sim = pd.read_csv(cluster+'_dir.sim.out',sep='\s+')
sim.rename(columns={'stage':'stage1','Cmprior':'CMprior'},inplace=True)
new_ph = sim.copy(deep=True)
for i in range(len(filters)):
    for j in range(len(partitions[i])-1):
        sim_bin = new_ph.loc[(partitions[i][j] <= new_ph[filters[i]]) & (new_ph[filters[i]] < partitions[i][j+1])].index
        if len(sim_bin) == 0:
            continue
        else:
            if scatters[i][j]  == 0:
                noise = [0.02]*len(sim_bin)
            else:
                noise = np.random.normal(means[i][j], scatters[i][j],len(sim_bin))
            for l in range(len(sim_bin)):
                offset = np.random.normal(0, np.abs(noise[l]))
                new_ph.loc[sim_bin[l],filters[i]] = new_ph.loc[sim_bin[l],filters[i]]+offset
                new_ph.loc[sim_bin[l],'sig'+filters[i]] = max(0.02, np.abs(noise[l]))
index = new_ph[ (new_ph['G'] <= 20) & (new_ph['G'] >= min_mag)].index
drops = [i for i in range(len(new_ph)) if i not in index]
new_ph.drop(drops, inplace=True)
for j in filters:
    new_ph.loc[new_ph['sig'+j]==0, 'sig'+j] = -9.9
new_ph[cols][:length].to_csv('simCluster.phot',sep=' ', index=False)
