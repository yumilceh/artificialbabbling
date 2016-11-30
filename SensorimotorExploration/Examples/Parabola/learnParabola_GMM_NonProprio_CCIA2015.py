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
    from Algorithm.Algorithm_CCIA2015 import Algorithm_CCIA2015 as Algorithm
    from Algorithm.Algorithm_CCIA2015 import MODELS
    from Models.ILGMM_SM import GMM_SM
    from Models.GMM_SS import GMM_SS
    from Models.GMM_IM import GMM_IM
    from Algorithm.ModelEvaluation import SM_ModelEvaluation
    from DataManager.PlotTools import *
   
    ## Simulation Parameters ##
    n_initialization=200
    n_evaluation_samples=100
    n_experiments=200
    random_seed=1234
    
    k_sm = 3
    sm_step=100
    alpha_sm=0.05
    
    k_ss = 6
    ss_step=100
    alpha_ss=0.05
    
    k_im=10
    im_step=20
    im_samples=800
    
    # To guarantee reproductible experiments##
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
    
    models.f_im = GMM_IM(system,
                         k_im,
                         n_training_samples=im_samples,
                         im_step=im_step)

    ## Creating Simulation object, running simulation and plotting experiments##
    file_prefix='Parabolic_NP_GMM_'
    simulation1=Algorithm(system,
                          models,
                          file_prefix=file_prefix,
                          n_experiments = n_experiments,
                          n_initialization_experiments = n_initialization,
                          g_im_initialization_method = 'non-painful',
                          n_save_data=100
                          )
    

    simulation1.runProprioceptiveAlgorithm()
    
    initialization_data_sm_ss=simulation1.data.initialization_data_sm_ss
    initialization_data_im=simulation1.data.initialization_data_im
    simulation_data=simulation1.data.simulation_data
    
    
    fig1,ax1=initializeFigure();
    fig1,ax1=simulation_data.plotSimulatedData2D(fig1,ax1,'sensor', 0, 'sensor', 1,"or")
    
    
    ## Validation of the model ##
    n_samples=n_evaluation_samples
    evaluation=SM_ModelEvaluation(system,
                                  n_samples,
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
    
    fig5,ax5=initializeFigure();
    fig5,ax5=initialization_data_sm_ss.plotSimulatedData2D(fig5,ax5,'sensor', 0, 'sensor', 1,"or")
    fig5, ax5 = simulation1.models.f_sm.model.plotGMMProjection(fig5,ax5,2, 3)
    ax5.relim()
    ax5.autoscale_view()
    
    fig6, ax6=initializeFigure();
    fig6, ax6=initialization_data_im.plotSimulatedData2D(fig6,ax6,'sensor_goal', 0, 'sensor_goal', 1,"ob")
    plt.hold(True)
    fig6, ax6=simulation_data.plotSimulatedData2D(fig6,ax6,'sensor_goal', 0, 'sensor_goal', 1,"or")

    fig6, ax = simulation1.models.f_im.model.plotGMMProjection(fig6,ax6,1, 2)
    ax6.relim()
    ax6.autoscale_view()    
    
    
    fig9, ax9 = initializeFigure();
    fig9, ax9 = simulation_data.plotTemporalSimulatedData(fig9,ax9,'competence', 0,"r",moving_average=10)
    
    
    fig10, ax10 = initializeFigure();
    fig10, ax10 = validation_valSet_data.plotTemporalSimulatedData(fig10,ax10,'error_log', 0,"--b",moving_average=10)
    fig10, ax10 = validation_valSet_data.plotTemporalSimulatedData(fig10,ax10,'error', 0,"r",moving_average=10)
    

    plt.draw()
    plt.pause(0.001)
    try:
        str_opt = raw_input("Press [enter] to continue or [H + ENTER] to keep plots.")
        if str_opt == 'H':
            plt.show()
    except SyntaxError:
        pass
    
    
    
        