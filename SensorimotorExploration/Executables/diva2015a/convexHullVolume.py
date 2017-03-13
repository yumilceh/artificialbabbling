'''
Created on Oct 5, 2016

@author: Juan Manuel Acevedo Valle
'''


import os,sys
import numpy as np
import h5py
from scipy.spatial import ConvexHull
#=======================================================================================
#
# Experiment report and videos
#
#=======================================================================================
this_dir =  os.getcwd()

if __name__ == '__main__' and True:
     
    #------------------------------------------------- sys.path.append("../../")
    #------- from SensorimotorSystems.DivaProprio2015a import DivaProprio2015a
    
    diva_output_scale=[100.0,500.0,1500.0,3000.0]

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
                   
    hull_volumes = {key: None for key in directories}
    
    for directory in directories:    
        mat = h5py.File(directory + 'SMdata.mat','r')
        data = np.array(mat.get('SMdata'))
        motor_data = data[6:,:]
        sensor_data = data[[0,1,3,4],:]
        print('Working on ' + directory)
         
        hull = ConvexHull(np.transpose(sensor_data))
        
        hull_volumes[directory]=hull.volume
       
        
    for keys,values in hull_volumes.items():
        print(keys)
        print(values)
        
        