'''
Created on Feb 5, 2016

@author: Juan Manuel Acevedo Valle
'''
import sys, os
import random    
        
if __name__ == '__main__':
    random.seed(1234)
    
    #Adding required paths
    print(os.getcwd())
    sys.path.append(os.getcwd()) 
    
    from SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a
    from Models.GeneralModels.GMM_PBD import GMM
    from Algorithm.StorageDataFunctions import loadSimulationData

    
    #Creating agent
    agent = Diva_Proprio2015a();
       
    #Loading data
    simulation_results = loadSimulationData('simulation_data_1stAttempt.tar.gz', agent)
    data = simulation_results['simulation_data']
    
    data=data.sensor_data.data.iloc[0:400]

    
    GMM_test=GMM(10)
    
    GMM_test.initializeMixture(data)
     
    GMM_test.train(data)

    