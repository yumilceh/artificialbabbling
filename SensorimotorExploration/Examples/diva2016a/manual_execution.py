'''
Created on Feb 5, 2016

@author: yumilceh
'''
from numpy import linspace
from numpy import random as np_rnd

if __name__ == '__main__':
   
     
    ## Adding the projects folder to the path##
    import os,sys,random
    sys.path.append(os.getcwd())

    ## Adding libraries##
    from SensorimotorSystems.Diva_Proprio2016a import Diva_Proprio2016a as System

    random_seed=1234
    n_experiments=200

    ## To guarantee reproductible experiments##
    random.seed(random_seed)
    np_rnd.seed(random_seed)

    ## Creating Agent ##
    system=System()
    
    ## Running interactive simulation
    file_prefix='Manual_Simulation'
    
    system.interactiveSystem()
    
    
    
    

    
