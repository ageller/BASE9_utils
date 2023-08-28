#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd


# In[69]:


clusterName='NGC_188'
df = pd.read_csv('Run1/'+clusterName+'_dir.res',sep='\s+')


# In[70]:


for i in range(2,11):
    df1 = pd.read_csv('Run'+str(i)+'/'+clusterName+'_dir.res',sep='\s+')
    df = pd.concat([df, df1[df1['stage']==3]], ignore_index=True)


# In[71]:


df.to_csv(clusterName+'.res',index=False,sep='\t')

