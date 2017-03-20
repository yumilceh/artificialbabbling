
# coding: utf-8

# In[9]:

'''
Created on Oct 5, 2016

@author: Juan Manuel Acevedo Valle
'''
import os,sys
import numpy as np
import h5py
from scipy.spatial import ConvexHull
this_dir =  os.getcwd()


# In[10]:

only_nocollisions = True

os.chdir('../ExperimentsIEEETCDS2016/')
directories = ['EVD_no_Proprio_0/',
               'EVD_no_Proprio_1/',
               'EVD_no_Proprio_2/',
               'EVD_no_Proprio_3/',
               'EVD_no_Proprio_4/',
               'EVD_no_Proprio_6/',
               'EVD_no_Proprio_7/',
               'EVD_no_Proprio_8/',
               'EVD_no_Proprio_9/',
               'EVD_Proprio_0/',
               'EVD_Proprio_1/',
               'EVD_Proprio_2/',
               'EVD_Proprio_3/',
               'EVD_Proprio_4/',
               'EVD_Proprio_6/',
               'EVD_Proprio_7/',
               'EVD_Proprio_8/',
               'EVD_Proprio_9/',
               'Special_EVD_Proprio_5/EVD_no_Proprio_5/',
               'Special_EVD_Proprio_5/EVD_Proprio_5/']
diva_output_scale=[100.0,500.0,1500.0,3000.0]


# In[12]:

n_directories = len(directories)
hull_volumes = {key: None for key in directories}

for i in range(n_directories):    
    mat = h5py.File(directories[i] + 'SMdata.mat','r')
    data = np.array(mat.get('SMdata'))
    #motor = data[6:,:]
    sensor_data = data[[0,1,3,4],:]
    
    print('Working on directory {} of {}'.format(i+1,n_directories) )
    hull = ConvexHull(np.transpose(sensor_data))
    hull_volumes[directories[i]]=[hull.volume, 0]
    
    if only_nocollisions:
        mat = h5py.File(directories[i] + 'PRdata.mat','r')
        data = np.array(mat.get('PRdata'))
        proprio_data = data[[-1],:]
        #print(proprio_data)
        #print(proprio_data.shape)
        #print(np.where(proprio_data == 0.))[1]

        sensor_data = sensor_data[:,np.where(proprio_data == 0)[1]]
        hull = ConvexHull(np.transpose(sensor_data))
        hull_volumes[directories[i]][1]=hull.volume
        #print(sensor)
    


# In[13]:

for key,value in hull_volumes.items():
    print(key + ":           " + str(value))


# In[21]:

os.chdir(this_dir)
### THAT'S IT OLD SPORT ###

