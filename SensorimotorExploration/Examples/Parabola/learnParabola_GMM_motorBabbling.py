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
    from Algorithm.Algorithm_Random import Algorithm_Random as Algorithm
    from Algorithm.Algorithm_Random import MODELS 
    from Models.GMM_SM import GMM_SM 
    from Models.GMM_SS import GMM_SS
    from Algorithm.ModelEvaluation import SM_ModelEvaluation
    from DataManager.PlotTools import *
   
    ## Simulation Parameters ##
    n_initialization=20
    n_evaluation_samples=100
    n_experiments=200
    random_seed=1234
    
    k_sm = 10
    sm_step=100
    alpha_sm=0.1
    
    k_ss = 10
    ss_step=100
    alpha_ss=0.1

    ## To guarantee reproductible experiments##
    random.seed(random_seed)
    np_rnd.seed(random_seed)

    ## Creating Agent ##
    system=System()
    
    ## Creating Models ##
    models=MODELS()
    
    models.f_sm = GMM_SM(system,
                         k_sm,
                         sm_step=sm_step,
                         alpha=alpha_sm)
    models.f_ss = GMM_SS(system,
                         k_ss,
                         ss_step=ss_step,
                         alpha=alpha_ss)

    ## Creating Simulation object, running simulation and plotting experiments##
    file_prefix='Sinus_GMM_'
    simulation1=Algorithm(system,
                          models,
                          file_prefix=file_prefix,
                          n_experiments = n_experiments
                          )
    

    simulation1.runNonProprioceptiveAlgorithm()
    
    initialization_data_sm_ss=simulation1.data.initialization_data_sm_ss
    initialization_data_im=simulation1.data.initialization_data_im
    simulation_data=simulation1.data.simulation_data

    fig1,ax1=initializeFigure();
    fig1,ax1=simulation_data.plotSimulatedData2D(fig1,ax1,'sensor', 0, 'sensor', 1,"or")
    
    
    ## Validation of the model ##
    evaluation=SM_ModelEvaluation(system,
                                  n_evaluation_samples,
                                  simulation1.models.f_sm,
                                  file_prefix=file_prefix)
    evaluation.setValidationEvaluationSets()
    
    validation_valSet_data = evaluation.evaluateModel(saveData=True)    
    
    fig1, ax1 = validation_valSet_data.plotSimulatedData2D(fig1,ax1,'sensor', 0, 'sensor', 1,"ob")    
    #fig1, ax1 = validation_valSet_data.plotSimulatedData2D(fig1,ax1,'motor', 0, 'sensor_goal', 0,"ok")
    fig1, ax1 = simulation1.models.f_sm.model.plotGMMProjection(fig1,ax1,2, 3)
    ax1.relim()
    ax1.autoscale_view()
    
    
    fig2,ax2=initializeFigure();
    fig2,ax2=simulation_data.plotSimulatedData2D(fig2,ax2,'motor', 0, 'motor', 1,"or")
    
    fig2, ax2 = validation_valSet_data.plotSimulatedData2D(fig2,ax2,'motor', 0, 'motor', 1,"ob")    
    #fig1, ax1 = validation_valSet_data.plotSimulatedData2D(fig1,ax1,'motor', 0, 'sensor_goal', 0,"ok")
    fig2, ax2 = simulation1.models.f_sm.model.plotGMMProjection(fig2,ax2,0, 1)
    ax2.relim()
    ax2.autoscale_view()
    
    fig3,ax3=initializeFigure();
    fig3,ax3=simulation_data.plotSimulatedData2D(fig3,ax3,'motor', 0, 'sensor', 0,"or")
    
    fig3, ax3 = validation_valSet_data.plotSimulatedData2D(fig3,ax3,'motor', 0, 'sensor', 0,"ob")    
    #fig1, ax1 = validation_valSet_data.plotSimulatedData2D(fig1,ax1,'motor', 0, 'sensor_goal', 0,"ok")
    fig3, ax3 = simulation1.models.f_sm.model.plotGMMProjection(fig3,ax3,0, 2)
    ax3.relim()
    ax3.autoscale_view()
    
    fig4,ax4=initializeFigure();
    fig4,ax4=simulation_data.plotSimulatedData2D(fig4,ax4,'motor', 1, 'sensor', 1,"or")
    
    fig4, ax4 = validation_valSet_data.plotSimulatedData2D(fig4,ax4,'motor', 1, 'sensor', 1,"ob")    
    #fig1, ax1 = validation_valSet_data.plotSimulatedData2D(fig1,ax1,'motor', 0, 'sensor_goal', 0,"ok")
    fig4, ax4 = simulation1.models.f_sm.model.plotGMMProjection(fig4,ax4,1, 3)
    ax4.relim()
    ax4.autoscale_view()
    
    fig10, ax10 = initializeFigure();
    fig10, ax10 = validation_valSet_data.plotTemporalSimulatedData(fig10,ax10,'competence', 0,"r",moving_average=10)
    

    
    plt.show();