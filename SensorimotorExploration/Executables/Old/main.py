'''
Created on Feb 5, 2016

@author: yumilceh
'''

if __name__ == '__main__':
    import os,sys,random
     
    print(os.getcwd())
    sys.path.append(os.getcwd())

    from SensorimotorSystems.Sinus_Agent import Sinus_Agent
    from Algorithm.Algorithm_Random import Algorithm_Random, MODELS 
    from Models.GMM_SM import GMM_SM
    from Models.GMM_SS import GMM_SS
    from Models.GMM_IM import GMM_IM
    from DataManager.PlotTools import *
   
    random.seed(1234)
    
    k_sm = 30
    k_ss = 28
    k_im = 60

     
    agent=Sinus_Agent()
    
    models=MODELS()
    
    models.f_sm = GMM_SM(agent,k_sm)
    models.f_ss = GMM_SS(agent,k_ss)
    models.f_im = GMM_IM(agent,k_im)
    
    simulation1=Algorithm_Random(agent,
                           models,
                           file_prefix='Sinus_GMM_',
                           n_experiments = 10000
                           )
    
    simulation1.runNonProprioceptiveAlgorithm()
    
    initialization_data_sm_ss=simulation1.data.initialization_data_sm_ss
    initialization_data_im=simulation1.data.initialization_data_im
    simulation_data=simulation1.data.simulation_data

    h,ax3=initializeFigure();
    h,ax3=simulation_data.plotSimulatedData2D(h,ax3,'sensor', 0, 'motor', 0,"or")
     
    fig2,ax2=initializeFigure();
    fig2,ax2=simulation_data.plotTemporalSimulatedData(fig2,ax2,'competence', 0,"r",moving_average=200)
    
    plt.show();
    
    
    #Testing unfinished simulation  
#===============================================================================
#     from SensorimotorSystems.DivaProprio2015a import DivaProprio2015a
#     from Algto
#     from DataManager.PlotTools import *
# 
#     initialization_data_sm_ss=simulation1.initialization_data_sm_ss
#     initialization_data_im=simulation1.initialization_data_im
#     simulation_data=simulation1.simulation_data
#===============================================================================


    