'''
Created on Feb 5, 2016

@author: yumilceh
'''

if __name__ == '__main__':
    import os,sys
     
    print(os.getcwd())
    sys.path.append(os.getcwd())

    from SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a
    from Algorithm.Algorithm1 import Algorithm1  
    from DataVisualization.PlotTools import *
   
    
    diva_agent=Diva_Proprio2015a()
    
    simulation1=Algorithm1(diva_agent)
    
    simulation1.runNonProprioceptiveAlgorithm()
    
    initialization_data_sm_ss=simulation1.initialization_data_sm_ss
    initialization_data_im=simulation1.initialization_data_im
    simulation_data=simulation1.simulation_data
         
    initialization_data_sm_ss.to_pickle('initialization_sm_ss.p')
    initialization_data_im.to_pickle('initialization_im.p')     
    simulation_data.to_pickle('simulation.p')

    h,ax3=initializeFigure();
    h,ax3=simulation_data.plotSimulatedData2D(h,ax3,'sensor', 0, 'sensor', 3,"or")
    
    j,ax4=initializeFigure();
    j,ax4=simulation_data.plotTemporalSimulatedData(j,ax4,'competence', 0,"r")
    
    plt.show();
    
    
    
