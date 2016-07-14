'''
Created on Feb 5, 2016

@author: yumilceh
'''

if __name__ == '__main__':
    import os,sys,random
     
    print(os.getcwd())
    sys.path.append(os.getcwd())

    from SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a
    from Algorithm.Algorithm1 import Algorithm1  
    from DataManager.PlotTools import *
   
    random.seed(1234)
    
    
    diva_agent=Diva_Proprio2015a()
    
    simulation1=Algorithm1(diva_agent)
    
    simulation1.runNonProprioceptiveAlgorithm()
    
    initialization_data_sm_ss=simulation1.initialization_data_sm_ss
    initialization_data_im=simulation1.initialization_data_im
    simulation_data=simulation1.simulation_data

    h,ax3=initializeFigure();
    h,ax3=simulation_data.plotSimulatedData2D(h,ax3,'sensor', 0, 'sensor', 3,"or")
    
    j,ax4=initializeFigure();
    j,ax4=simulation_data.plotTemporalSimulatedData(j,ax4,'competence', 0,"r",moving_average=5000)
    
    plt.show();
    
    
    #Testing unfinished simulation  
#===============================================================================
#     from SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a
#     from Algto
#     from DataManager.PlotTools import *
# 
#     initialization_data_sm_ss=simulation1.initialization_data_sm_ss
#     initialization_data_im=simulation1.initialization_data_im
#     simulation_data=simulation1.simulation_data
#===============================================================================


    