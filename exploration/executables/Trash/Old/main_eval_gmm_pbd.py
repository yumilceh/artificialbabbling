'''
Created on Feb 5, 2016

@author: Juan Manuel Acevedo Valle
'''
import sys, os
import matplotlib.pyplot as plt
import random    
from numpy import random as np_rnd

      
if __name__ == '__main__':
    random.seed(1234)
    np_rnd.seed(1234)
    
    #Adding required paths
    print(os.getcwd())
    sys.path.append(os.getcwd()) 
    #sys.path.append('/home/yumilceh/Documents/IMLE/build/lib/')
    
    
    from SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a
    from Models.GMMpbd_SM import GMM_SM
    from Algorithm.StorageDataFunctions import loadSimulationData
    from Algorithm.ModelEvaluation import SM_ModelEvaluation
    from DataManager.PlotTools import initializeFigure

    
    #Creating agent
    agent = Diva_Proprio2015a();
       
    #Loading data
    simulation_results = loadSimulationData('simulation_data_1stAttempt.tar.gz', agent)
    data = simulation_results['simulation_data']
    
    #Initializing IMLE model
    model = GMM_SM(agent, 28)
    
    #Generating Training and Validation Sets
    ratio_samples_val=0.2
    saveData = True
    eva_train_set = 0.2
    
    model_eval=SM_ModelEvaluation(agent,data,model,ratio_samples_val)
    validation_trainSet_data, validation_valSet_data = model_eval.evaluate(saveData, eva_train_set)
    
    fig1,ax1=initializeFigure();
    fig1,ax1=validation_trainSet_data.plot_time_series(fig1, ax1, 'competence', 0, "r", moving_average=2000)
    
    
    fig2,ax2=initializeFigure();
    fig2,ax2=validation_valSet_data.plot_time_series(fig2, ax2, 'competence', 0, "r", moving_average=2000)
    
    plt.show();