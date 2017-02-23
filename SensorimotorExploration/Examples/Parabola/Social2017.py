'''
Created on Feb 5, 2016

@author: Juan Manuel Acevedo Valle
'''
from numpy import linspace
from numpy import random as np_rnd
import datetime

now = datetime.datetime.now().strftime("Social_%Y_%m_%d_%H_%M_")

if __name__ == '__main__':
    #  Adding the projects folder to the path##
    import os, sys, random

    sys.path.append("../../")

    #  Adding libraries##
    from SensorimotorExploration.SensorimotorSystems.Parabola import ConstrainedParabolicArea as System
    from SensorimotorExploration.Algorithm.Social2017 import Social as Algorithm
    from SensorimotorExploration.Algorithm.Social2017 import OBJECT
    from SensorimotorExploration.Algorithm.ModelEvaluation import SM_ModelEvaluation
    from SensorimotorExploration.DataManager.PlotTools import *
    from SensorimotorExploration.Algorithm.utils.CompetenceFunctions import comp_Moulin2013_expl as comp_func_expl
    from SensorimotorExploration.Algorithm.utils.CompetenceFunctions import comp_Moulin2013 as comp_func

    from model_configurations import model_

    # Models
    f_sm_key = 'explauto_sm'
    f_ss_key = 'explauto_ss'
    f_im_key = 'explauto_im'

    '''
       'gmm_sm':  GMM_SM,
       'gmm_ss':  GMM_SS,
       'igmm_sm': IGMM_SM,
       'igmm_ss': IGMM_SS,
       'gmm_im':  GMM_IM,
       'explauto_im': ea_IM,
       'explauto_sm': ea_SM,
       'explauto_ss': ea_SS,
       'random':  RdnM
    '''

    # To guarantee reproducible experiments
    random_seed = 1234

    n_initialization = 35
    n_experiments = 400
    n_save_data = 200

    eval_step = 100

    random.seed(random_seed)
    np_rnd.seed(random_seed)

    # Creating Agent ##
    system = System()

    # Creating Models ##
    models = OBJECT()

    models.f_sm = model_(f_sm_key, system)
    models.f_ss = model_(f_ss_key, system)
    models.f_im = model_(f_im_key, system, competence_func=comp_func_expl)

    evaluation_sim = SM_ModelEvaluation(system,
                                        models.f_sm, comp_func=comp_func)

    evaluation_sim.loadEvaluationDataSet('parabola_validation_data_set_2.h5')

    #  Creating Simulation object, running simulation and plotting experiments##
    file_prefix = 'Parabola_Sim_' + now
    simulation = Algorithm(system,
                           models,
                           n_experiments,
                           comp_func,
                           n_initialization_experiments=n_initialization,
                           random_seed=1234,
                           g_im_initialization_method='all',
                           n_save_data=n_save_data,
                           evaluation=evaluation_sim,
                           eval_step = eval_step,
                           sm_all_samples=False)

    simulation.run_simple()


    init_data_sm = simulation.data.initialization_data_sm_ss
    init_data_im = simulation.data.initialization_data_im
    sim_data = simulation.data.simulation_data
    
    ## Validation of the model ##
    evaluation = SM_ModelEvaluation(system,
                                  simulation.models.f_sm)

    evaluation.loadEvaluationDataSet('parabola_validation_data_set_2.h5')
    
    validation_valSet_data = evaluation.evaluateModel(saveData=True)   
    
    fig1,ax1 = initializeFigure()
    fig1.suptitle('All Sensory Results')
    fig1, ax1 = init_data_sm.plotSimulatedData2D(fig1,ax1,'sensor', 0, 'sensor', 1,"ok")
    fig1, ax1 = init_data_im.plotSimulatedData2D(fig1,ax1,'sensor', 0, 'sensor', 1,"og")
    fig1,ax1 = sim_data.plotSimulatedData2D(fig1,ax1,'sensor', 0, 'sensor', 1,"or")
    fig1, ax1 = validation_valSet_data.plotSimulatedData2D(fig1,ax1,'sensor', 0, 'sensor', 1,"ob")    
    #fig1, ax1 = simulation.models.f_sm.model.plotGMMProjection(fig1,ax1,2, 3)
    ax1.relim()
    ax1.autoscale_view()

    plt.draw()
    plt.pause(0.001)
    try:
        str_opt = raw_input("Press [enter] to continue or [H + ENTER] to keep plots.")
        if str_opt == 'H':
            plt.show()
    except SyntaxError:
        pass

    '''
    fig2,ax2=initializeFigure()
    fig2.suptitle('Motor Commands: M1 vs M2')
    fig2,ax2=simulation_data.plotSimulatedData2D(fig2,ax2,'motor', 0, 'motor', 1,"or")
    fig2, ax2 = validation_valSet_data.plotSimulatedData2D(fig2,ax2,'motor', 0, 'motor', 1,"ob")    
    fig2, ax2 = simulation1.models.f_sm.model.plotGMMProjection(fig2,ax2,0, 1)
    ax2.relim()
    ax2.autoscale_view()
    
    fig3,ax3=initializeFigure()
    fig3.suptitle('RESULTS: M1 vs S1')
    fig3,ax3=simulation_data.plotSimulatedData2D(fig3,ax3,'motor', 0, 'sensor', 0,"or")
    fig3, ax3 = validation_valSet_data.plotSimulatedData2D(fig3,ax3,'motor', 0, 'sensor', 0,"ob")    
    fig3, ax3 = simulation1.models.f_sm.model.plotGMMProjection(fig3,ax3,0, 2)
    ax3.relim()
    ax3.autoscale_view()
    
    fig4,ax4=initializeFigure()
    fig4.suptitle('RESULTS: M2 vs S2')
    fig4,ax4=simulation_data.plotSimulatedData2D(fig4,ax4,'motor', 1, 'sensor', 1,"or")   
    fig4, ax4 = validation_valSet_data.plotSimulatedData2D(fig4,ax4,'motor', 1, 'sensor', 1,"ob")    
    fig4, ax4 = simulation1.models.f_sm.model.plotGMMProjection(fig4,ax4,1, 3)
    ax4.relim()
    ax4.autoscale_view()
    
    fig5,ax5=initializeFigure()
    fig5.suptitle('Initialization data: S1 vs S2')
    fig5,ax5=initialization_data_sm_ss.plotSimulatedData2D(fig5,ax5,'sensor', 0, 'sensor', 1,"or")
    fig5, ax5 = simulation1.models.f_sm.model.plotGMMProjection(fig5,ax5,2, 3)
    ax5.relim()
    ax5.autoscale_view()
    
    fig6, ax6=initializeFigure()
    fig6.suptitle('Inialization data: S_g1 vs S_g2')
    fig6, ax6=initialization_data_im.plotSimulatedData2D(fig6,ax6,'sensor_goal', 0, 'sensor_goal', 1,"ob")
    plt.hold(True)
    fig6, ax6=simulation_data.plotSimulatedData2D(fig6,ax6,'sensor_goal', 0, 'sensor_goal', 1,"or")
    fig6, ax = simulation1.models.f_im.model.plotGMMProjection(fig6,ax6,1, 2)
    ax6.relim()
    ax6.autoscale_view()    
    
    
    fig7, ax7 =  initializeFigure();
    fig7.suptitle('Evaluation Error Evolution')
    plt.plot(simulation1.evaluation_error[1:],'b')
    plt.hold(True)
    plt.xlabel('Sensorimotor training step')
    plt.ylabel('Mean error') 
    
    
    
    fig8, ax8 =  initializeFigure();
    fig8.suptitle('Validation: S1 vs S2')
    fig8, ax8 = validation_valSet_data.plotSimulatedData2D(fig8, ax8,'sensor', 0, 'sensor', 1,"ob")
    plt.hold(True)
    fig8, ax8 = validation_valSet_data.plotSimulatedData2D(fig8,ax8,'sensor_goal', 0, 'sensor_goal', 1,"or")
    ax8.legend(['Results','Goal'])
           
    fig9, ax9 = initializeFigure();
    fig9.suptitle('Competence during Training')
    fig9, ax9 = simulation_data.plotTemporalSimulatedData(fig9,ax9,'competence', 0,"r",moving_average=10)
    
    fig10, ax10 = initializeFigure();
    fig10.suptitle('Competence and Error during validation')
    fig10, ax10 = validation_valSet_data.plotTemporalSimulatedData(fig10,ax10,'competence', 0,"--b",moving_average=10)
    fig10, ax10 = validation_valSet_data.plotTemporalSimulatedData(fig10,ax10,'error', 0,"r",moving_average=10)
    
    
    
    


'''
