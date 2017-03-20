'''
Created on Jun 21, 2016

@author: Juan Manuel Acevedo Valle
'''

'''
Created on Feb 5, 2016

@author: yumilceh
'''

if __name__ == '__main__':
    import os,sys,random
     
    print(os.getcwd())
    sys.path.append(os.getcwd())    
    
    #Testing unfinished simulation  
    from SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a
    from Algorithm.StorageDataFunctions import loadSimulationData
    from DataManager.PlotTools import *

    random.seed(1234)
    agent=Diva_Proprio2015a();

    simulation_results=loadSimulationData('validation_results.tar.gz', agent)
    validation_trainSet_data=simulation_results['validation_trainSet_data']
    validation_valSet_data=simulation_results['validation_valSet_data']
    
    
    h1,ax1=initializeFigure();
    h1,ax1=validation_trainSet_data.plot_time_series(h1, ax1, 'competence', 0, "r", moving_average=2000)
    ax1.set_ylabel('Average Competence (2000 samples MA)')
    ax1.set_xlabel('Vocalization [k]')
    ax1.set_title('Evaluation with the Training set')
       
    h2,ax2=initializeFigure();
    h2,ax2=validation_valSet_data.plot_time_series(h2, ax2, 'competence', 0, "r", moving_average=2000)
    ax2.set_ylabel('Average Competence (2000 samples MA)')
    ax2.set_xlabel('Vocalization [k]')
    ax2.set_title('Evaluation with theValidation set')

    plt.show();