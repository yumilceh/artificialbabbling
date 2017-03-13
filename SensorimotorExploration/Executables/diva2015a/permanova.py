'''
Created on Oct 10, 2016

@author: Juan Manuel Acevedo Valle
''' 
from skbio import DistanceMatrix
from skbio.stats.distance import anosim, permanova
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
import h5py, os, sys, random, time
    
if __name__ == '__main__':
    
    this_dir =  os.getcwd()
    
    dist_used = 'euclidean'
    n_rnd_samples = 5000
    n_permutations = 30
    n_iterations = 10
    random.seed(1234)
    np.random.seed(1234)
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
    #--------------------------------------- directories = ['EVD_no_Proprio_0/',
                   #--------------------------------------- 'EVD_no_Proprio_1/',
                   #------------------------------------------ 'EVD_Proprio_0/']

    test_statistic_values = np.zeros((len(directories),len(directories),n_iterations))
    p_values = np.zeros((len(directories),len(directories),n_iterations))
    t0 = time.clock()
    for k in range(n_iterations):
        indices1 = np.array(random.sample(range(499999),n_rnd_samples))
        indices2 = np.array(random.sample(range(499999),n_rnd_samples))

        for i in range(len(directories)): 
            directory1 = directories[i]
            mat1 = h5py.File(directory1 + 'SMdata.mat','r')
            sensor_data1 = np.array(mat1.get('SMdata'))
            sensor_data1 = np.transpose(sensor_data1[[0,1,3,4],:])
            sensor_data1 = sensor_data1[indices1,:]
            print('Working on ' + directory1 )
            for j in range(i, len(directories)):
                directory2 = directories[j]
                mat2 = h5py.File(directory2 + 'SMdata.mat','r')
                sensor_data2 = np.array(mat2.get('SMdata'))
                sensor_data2 = np.transpose(sensor_data2[[0,1,3,4],:])
                sensor_data2 = sensor_data2[indices2,:]
                sensor_data = np.append(sensor_data1,sensor_data2,axis=0)
                distances = pdist(sensor_data, dist_used)
                distances = squareform(distances)
                
                #----- #Iterative computation of dm but slow
                #----- distance_matrix = np.zeros((n_rnd_samples,n_rnd_samples))
                #------------------------------- for ii in range(n_rnd_samples):
                    #--------------------------- for jj in range(n_rnd_samples):
                        # dist = pdist(np.append([sensor_data1[ii,:]],[sensor_data2[jj,:]], axis = 0), dist_used)
                        #------------------------- distance_matrix[ii,jj] = dist
                        #------------------------- distance_matrix[jj,ii] = dist
                        
                distances = DistanceMatrix(distances)
                grouping = np.append(['GMM1']*n_rnd_samples,['GMM2']*n_rnd_samples, axis=0)
                pnv_tmp = permanova(distances, grouping, permutations = n_permutations)

                test_statistic_values[i,j,k] = pnv_tmp['test statistic']
                p_values[i,j,k] = pnv_tmp['p-value']
    os.chdir(this_dir)
    np.save('permanova_result_ts.npy', test_statistic_values)
    np.save('permanova_result_p.npy', p_values)
    print(str(time.clock() - t0) + "seconds") 

