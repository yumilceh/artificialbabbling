'''
Created on Jun 21, 2016

@author: Juan Manuel Acevedo Valle
'''

'''
Created on Feb 5, 2016

@author: yumilceh
'''

if __name__ == '__main__':
    import os,sys
     
    print(os.getcwd())
    sys.path.append(os.getcwd())    
    
    #Testing unfinished simulation  
    from SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a
    from Algorithm.StorageDataFunctions import loadSimulationData
    from DataManager.PlotTools import *

    agent=Diva_Proprio2015a();

    simulation_results=loadSimulationData('simulation_data_1stAttempt.tar.gz', agent)
    initialization_data_sm_ss=simulation_results['initialization_data_sm_ss']
    initialization_data_im=simulation_results['initialization_data_im']
    simulation_data=simulation_results['simulation_data']
    
    h1,ax_h1=initializeFigure();
    h1,ax_h1=simulation_data.plotSimulatedData2D(h1,ax_h1,'sensor', 0, 'sensor', 1,"or")
    
    h2,ax_h2=initializeFigure();
    h2,ax_h2=simulation_data.plotSimulatedData2D(h2,ax_h2,'sensor', 3, 'sensor', 4,"or")
    
    h3,ax_h3=initializeFigure();
    h3,ax_h3=simulation_data.plotSimulatedData2D(h3,ax_h3,'sensor', 2, 'sensor', 5,"or")
    
    j,ax4=initializeFigure();
    j,ax4=simulation_data.plotTemporalSimulatedData(j,ax4,'competence', 0,"r",moving_average=5000)
    
    plt.show();