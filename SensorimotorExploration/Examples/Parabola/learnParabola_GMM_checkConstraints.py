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
    from SensorimotorSystems.Parabola import ConstrainedParabolicArea as System
    from Algorithm.InteractiveSimulation import ManualSimulation as Algorithm
    from Models.GMMpbd_SM import GMM_SM 
    from Models.GMM_SS import GMM_SS
    from Algorithm.ModelEvaluation import SM_ModelEvaluation
    from DataManager.PlotTools import *
   
    random_seed=1234
    n_experiments=200

    ## To guarantee reproductible experiments##
    random.seed(random_seed)
    np_rnd.seed(random_seed)

    ## Creating Agent ##
    system=System()
    

    ## Creating Simulation object, running simulation and plotting experiments##
    file_prefix='Sinus_GMM_'
    simulation1=Algorithm(system,
                          file_prefix=file_prefix,
                          n_experiments = n_experiments
                          )
    

    simulation1.executeManualMotorCommands()
    
    

    