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

    file_name='simulation_data_1stAttempt.tar.gz'
    
    simulation_results=loadSimulationData(file_name, agent)
    initialization_data_sm_ss=simulation_results['initialization_data_sm_ss']
    initialization_data_im=simulation_results['initialization_data_im']
    simulation_data=simulation_results['simulation_data']
    
    h1,ax1=initializeFigure();
    h1,ax1=simulation_data.plotSimulatedData2D(h1,ax1,'sensor', 0, 'sensor', 1,"or")
    ax1.set_ylabel('F_{21}')
    ax1.set_xlabel('F_{11}')
    ax1.set_title('First Perceptual Window Projection')
    
    h2,ax2=initializeFigure();
    h2,ax2=simulation_data.plotSimulatedData2D(h2,ax2,'sensor', 3, 'sensor', 4,"or")
    ax2.set_ylabel('F_{22}')
    ax2.set_xlabel('F_{12}')
    ax2.set_title('Second Perceptual Window Projection')
    
    h3,ax3=initializeFigure();
    h3,ax3=simulation_data.plotSimulatedData2D(h3,ax3,'sensor', 2, 'sensor', 5,"or")
    ax3.set_ylabel('I_2')
    ax3.set_xlabel('I_1')
    ax3.set_title('Intonation Parameter Projection')
    
    h4,ax4=initializeFigure();
    h4,ax4=simulation_data.plotTemporalSimulatedData(h4,ax4,'competence', 0,"r",moving_average=5000)
    ax4.set_ylabel('Average Competence (5000 samples MA)')
    ax4.set_xlabel('Vocalization [k]')
    ax4.set_title('Competence Evolution')
    
    plt.show();