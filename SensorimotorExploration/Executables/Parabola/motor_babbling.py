'''
Created on Feb 5, 2016

@author: yumilceh
'''
from numpy import linspace
from numpy import random as np_rnd
import pandas as pd

class OBJECT(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    ## Adding the projects folder to the path##
    import os,sys,random
    ## Adding libraries##
    from SensorimotorExploration.Systems.Parabola import ParabolicRegion as System
    from SensorimotorExploration.Algorithm.AlgorithmRandom import Algorithm_Random as Algorithm
    from SensorimotorExploration.Algorithm.AlgorithmRandom import MODELS
    from SensorimotorExploration.Algorithm.ModelEvaluation import SM_ModelEvaluation
    from SensorimotorExploration.DataManager.PlotTools import *

    from model_configurations import model_

    # Models
    f_sm_key = 'explauto_sm'
    f_ss_key = 'igmm_ss'
    f_im_key = 'explauto_im'

    """
       'gmm_sm':  GMM_SM,
       'gmm_ss':  GMM_SS,
       'igmm_sm': IGMM_SM,
       'igmm_ss': IGMM_SS,
       'gmm_im':  GMM_IM,
       'explauto_im': ea_IM,
       'explauto_sm': ea_SM,
       'explauto_ss': ea_SS,
       'random':  RdnM
    """

    ## Simulation Parameters ##
    n_initialization=200
    n_evaluation_samples=200
    n_experiments=200
    
    random_seed=1234
    
    k_sm = 30
    sm_step=100
    alpha_sm=0.05
    
    k_ss = 6
    ss_step=100
    alpha_ss=0.05

    ## To guarantee reproductible experiments##
    random.seed(random_seed)
    np_rnd.seed(random_seed)

    ## Creating Agent ##
    system=System()
    
    ## Creating Models ##
    models=OBJECT()


    """
        Write dictionary that allows to use any model.
    """
    models.f_sm = model_(f_sm_key, system)
    models.f_ss = model_(f_ss_key, system)
    models.f_im = model_(f_im_key, system)

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

    #fig1,ax1=initializeFigure();
    #fig1,ax1=simulation_data.plot_2D(fig1,ax1,'sensor', 0, 'sensor', 1,"or")
    
    
    ## Validation of the model ##
    n_samples=n_evaluation_samples
    evaluation = SM_ModelEvaluation(system,
                                    simulation1.models.f_sm)

    evaluation.loadEvaluationDataSet('parabola_dataset_2.h5')
    
    validation_valSet_data = evaluation.evaluateModel(saveData=True)   
    
    fig1,ax1 = initializeFigure()
    fig1.suptitle('All Sensory Results')
    #fig1, ax1 = initialization_data_sm_ss.plot_2D(fig1,ax1,'sensor', 0, 'sensor', 1,"ok")
    #fig1, ax1 = initialization_data_im.plot_2D(fig1,ax1,'sensor', 0, 'sensor', 1,"og")
    #fig1,ax1 = simulation_data.plot_2D(fig1,ax1,'sensor', 0, 'sensor', 1,"or")
    fig1, ax1 = validation_valSet_data.plot_2D(fig1, ax1, 'sensor_goal', 0, 'sensor_goal', 1, "ob")
    #fig1, ax1 = simulation1.models.f_sm.model.plotGMMProjection(fig1,ax1,2, 3)
    ax1.relim()
    ax1.autoscale_view()
    
    
    # fig2,ax2=initializeFigure()
    # fig2.suptitle('Motor Commands: M1 vs M2')
    # fig2,ax2=simulation_data.plot_2D(fig2,ax2,'motor', 0, 'motor', 1,"or")
    # fig2, ax2 = validation_valSet_data.plot_2D(fig2,ax2,'motor', 0, 'motor', 1,"ob")
    # fig2, ax2 = simulation1.models.f_sm.model.plotGMMProjection(fig2,ax2,0, 1)
    # ax2.relim()
    # ax2.autoscale_view()
    #
    # fig3,ax3=initializeFigure()
    # fig3.suptitle('RESULTS: M1 vs S1')
    # fig3,ax3=simulation_data.plot_2D(fig3,ax3,'motor', 0, 'sensor', 0,"or")
    # fig3, ax3 = validation_valSet_data.plot_2D(fig3,ax3,'motor', 0, 'sensor', 0,"ob")
    # fig3, ax3 = simulation1.models.f_sm.model.plotGMMProjection(fig3,ax3,0, 2)
    # ax3.relim()
    # ax3.autoscale_view()
    
    # fig4,ax4=initializeFigure()
    # fig4.suptitle('RESULTS: M2 vs S2')
    # fig4,ax4=simulation_data.plot_2D(fig4,ax4,'motor', 1, 'sensor', 1,"or")
    # fig4, ax4 = validation_valSet_data.plot_2D(fig4,ax4,'motor', 1, 'sensor', 1,"ob")
    # fig4, ax4 = simulation1.models.f_sm.model.plotGMMProjection(fig4,ax4,1, 3)
    # ax4.relim()
    # ax4.autoscale_view()
    
    # fig5,ax5=initializeFigure()
    # fig5.suptitle('Initialization data: S1 vs S2')
    # fig5,ax5=initialization_data_sm_ss.plot_2D(fig5,ax5,'sensor', 0, 'sensor', 1,"or")
    # fig5, ax5 = simulation1.models.f_sm.model.plotGMMProjection(fig5,ax5,2, 3)
    # ax5.relim()
    # ax5.autoscale_view()
    #
    #===========================================================================
    # fig6, ax6=initializeFigure()
    # fig6.suptitle('Inialization data: S_g1 vs S_g2')
    # fig6, ax6=initialization_data_im.plot_2D(fig6,ax6,'sensor_goal', 0, 'sensor_goal', 1,"ob")
    # plt.hold(True)
    # fig6, ax6=simulation_data.plot_2D(fig6,ax6,'sensor_goal', 0, 'sensor_goal', 1,"or")
    # fig6, ax = simulation1.models.f_im.model.plotGMMProjection(fig6,ax6,1, 2)
    # ax6.relim()
    # ax6.autoscale_view()    
    #===========================================================================
    
    
    #===========================================================================
    # fig2, ax7 =  initializeFigure();
    # fig2.suptitle('Evaluation Error Evolution')
    # plt.plot(simulation1.evaluation_error[1:],'b')
    # plt.hold(True)
    # plt.xlabel('Sensorimotor training step')
    # plt.ylabel('Mean error') 
    # 
    #===========================================================================
    
    #
    # fig3, ax3 =  initializeFigure();
    # fig3.suptitle('Validation: S1 vs S2')
    # fig3, ax3 = validation_valSet_data.plot_2D(fig3, ax3,'sensor', 0, 'sensor', 1,"ob")
    # plt.hold(True)
    # fig3, ax3 = validation_valSet_data.plot_2D(fig3,ax3,'sensor_goal', 0, 'sensor_goal', 1,"or")
    # ax3.legend(['Results','Goal'])
           
    fig9, ax9 = initializeFigure()
    fig9.suptitle('Competence during Training')
    fig9, ax9 = simulation_data.plot_time_series(fig9, ax9, 'competence', 0, "r", moving_average=10)
    
    fig10, ax10 = initializeFigure()
    fig10.suptitle('Competence and Error during validation')
    fig10, ax10 = validation_valSet_data.plot_time_series(fig10, ax10, 'competence', 0, "--b", moving_average=10)
    fig10, ax10 = validation_valSet_data.plot_time_series(fig10, ax10, 'error', 0, "r", moving_average=10)
    
    
    
    
    plt.draw()
    plt.show()


    