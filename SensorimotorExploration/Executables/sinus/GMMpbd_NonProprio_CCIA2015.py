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
    from SensorimotorSystems.Sinus import Sinus as System
    from Algorithm.Algorithm_CCIA2015 import Algorithm_CCIA2015 as Algorithm
    from Algorithm.Algorithm_CCIA2015 import MODELS
    from Models.GMMpbd_SM import GMM_SM
    from Models.GMM_SS import GMM_SS
    from Models.GMMpbd_IM import GMM_IM
    from Algorithm.ModelEvaluation import SM_ModelEvaluation
    from DataManager.PlotTools import *
   
    ## Simulation Parameters ##
    n_initialization=20
    n_evaluation_samples=100
    n_experiments=100
    random_seed=1234
    
    k_sm = 5
    sm_step=30
    alpha_sm=0.1
    
    k_ss = 5
    ss_step=30
    alpha_ss=0.1
    
    k_im=4
    im_step=20
    im_samples=80
    
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
    file_prefix='Sinus_GMM_'
    simulation1=Algorithm(system,
                          models,
                          file_prefix=file_prefix,
                          n_experiments = n_experiments,
                          n_initialization_experiments=n_initialization,
                          g_im_initialization_method = 'all',
                          n_save_data=100
                          )
    

    simulation1.runNonProprioceptiveAlgorithm()
    
    initialization_data_sm_ss=simulation1.data.initialization_data_sm_ss
    initialization_data_im=simulation1.data.initialization_data_im
    simulation_data=simulation1.data.simulation_data

    fig1,ax1=initializeFigure();
    fig1,ax1=simulation_data.plot_2D(fig1, ax1, 'motor', 0, 'sensor', 0, "or")
    
    
    ## Validation of the model ##
    evaluation=SM_ModelEvaluation(system,
                                  n_evaluation_samples,
                                  simulation1.models.f_sm,
                                  file_prefix=file_prefix)
    evaluation.setValidationEvaluationSets()
    
    validation_valSet_data = evaluation.evaluateModel(saveData=True)    
    
    fig1,ax1=validation_valSet_data.plot_2D(fig1, ax1, 'motor', 0, 'sensor', 0, "ob")
    fig1,ax1=validation_valSet_data.plot_2D(fig1, ax1, 'motor', 0, 'sensor_goal', 0, "ok")
    fig1,ax1 = simulation1.models.f_sm.model.plotGMMProjection(fig1,ax1,0, 1)
    ax1.relim()
    ax1.autoscale_view()
    
    fig3,ax3=initializeFigure();
    fig3,ax3 = simulation1.models.f_im.model.plotGMMProjection(fig3,ax3,0, 3)
    ax3.relim()
    ax3.autoscale_view()
 
    fig3a,ax3a = initializeFigure();
    fig3a,ax3a = simulation1.models.f_im.model.plotGMMProjection(fig3a,ax3a,1, 2)
    ax3a.relim()
    ax3a.autoscale_view()
     
    ## NOT WORKING WELL YET ##
    fig3b,ax3b=initializeFigure3D();
    fig3b,ax3b = simulation1.models.f_im.model.plotGMM3DProjection(fig3b,ax3b,2, 3, 0)
    ax3b.relim()
    ax3b.autoscale_view()
     

    fig2,ax2=initializeFigure();
    fig2,ax2=validation_valSet_data.plot_time_series(fig2, ax2, 'competence', 0, 'r', moving_average=0)

    plt.show();