'''
Created on Jan 23, 2017

@author: Juan Manuel Acevedo Valle
'''
if __name__ == '__main__':
    import numpy as np
    
    from exploration.models.sensorimotor.ExplautoSM import ExplautoSM
    from exploration.models.Interest.ExplautoIM import explauto_IM
    from exploration.data.data import SimulationData
    
    from exploration.systems.trash.Parabola import ParabolicRegion as System
    # from exploration.systems.Arm_explauto import SimpleArm as System
    # from systems.Diva_Proprio2016a import Diva_Proprio2016a as System

    from exploration.algorithm.utils.competence_funcs import comp_Moulin2013 as get_competence
    from exploration.algorithm.utils.competence_funcs import comp_Moulin2013_expl as competence
    
    from exploration.algorithm.utils.functions import get_random_motor_set, get_random_sensor_set
    from exploration.algorithm.trash.ModelEvaluation import SM_ModelEvaluation


    system = System()
    model_type = 'discretized_progress'  #'tree' 'discretized_progress''gmm_progress_beta'
    im_model = explauto_IM(system, competence,  model_type)
    
    sm_model = ExplautoSM(system, "nearest_neighbor")
    sigma_explo_ratio = 0.1
    sm_model.set_sigma_explo_ratio(sigma_explo_ratio)
    
    evaluation = SM_ModelEvaluation(system,
                                    sm_model)
    evaluation.loadEvaluationDataSet('../Parabola/parabola_dataset_2.h5')
    
    simulation_data = SimulationData(system)
    
    for m in get_random_motor_set(system, 10):
        system.set_action(m)
        system.executeMotorCommand()        
        simulation_data.append_data(system)
        sm_model.train(simulation_data)
        
    for s_g in get_random_sensor_set(system, 10):
        system.sensor_goal = s_g
        sm_model.get_action(system)
        system.executeMotorCommand()        
        simulation_data.append_data(system)
        sm_model.train(simulation_data)
        im_model.train(simulation_data)
    
    n_experiments = 400
    evaluation_samples = range(10,n_experiments+1, 1000)
    evaluation_error=[]
    
    for i in range(n_experiments):
        
        #---------------------------------------------------- #RANDOM EXPLORATION
        #-------------- system.sensor_goal = get_random_sensor_set(system, 1)[0]
        
        #------------------------------------------------------------- #CURIOSITY
        system.sensor_goal = im_model.get_goal(system)
        
        sm_model.get_action(system)
        system.executeMotorCommand() 
        get_competence(system)
        simulation_data.append_data(system)
        
        sm_model.train(simulation_data)
        im_model.train(simulation_data)
        if i in evaluation_samples:
            sm_model.set_sigma_explo_ratio(0)
            evaluation.model = sm_model
            eval_data = evaluation.evaluate_model()
            error_ = np.linalg.norm(eval_data.sensor_goal_data.data - eval_data.sensor_data.data, axis = 1)
            evaluation_error = np.append(evaluation_error, np.mean(error_))
            sm_model.set_sigma_explo_ratio(sigma_explo_ratio)
            
    
        #evaluate
    evaluation = SM_ModelEvaluation(system,
                                    sm_model)
    evaluation.loadEvaluationDataSet('../Parabola/parabola_dataset_2.h5')

    from exploration.data.PlotTools import *
    from matplotlib.pyplot import show
    
        
    fig2,ax2=initializeFigure()
    simulation_data.plot_2D(fig2, ax2, 'sensor', 0, 'sensor', 1, 'or')
    simulation_data.plot_2D(fig2, ax2, 'sensor_goal', 0, 'sensor_goal', 1, 'xk')
    
     
    fig3,ax3=initializeFigure()
    simulation_data.plot_time_series(fig3, ax3, 'competence', 0, 'b', moving_average=0)
    simulation_data.plot_time_series(fig3, ax3, 'competence', 0, 'r', moving_average=10)
    t_evaluation = np.asmatrix(np.array(evaluation_samples)) + 20.
    plt.plot(t_evaluation.tolist()[0], evaluation_error, 'k')

    sm_model.set_sigma_explo_ratio(0)
    evaluation = SM_ModelEvaluation(system,
                                    sm_model)

    evaluation.loadEvaluationDataSet('../Parabola/parabola_dataset_2.h5')

    validation_valSet_data = evaluation.evaluate_model(saveData=True)

    fig4, ax4 = initializeFigure()
    fig4.suptitle('All Sensory Results')
    fig4, ax4 = validation_valSet_data.plot_2D(fig4, ax4, 'sensor_goal', 0, 'sensor_goal', 1, "ob")
    ax4.relim()
    ax4.autoscale_view()

    fig10, ax10 = initializeFigure()
    fig10.suptitle('Competence and Error during validation')
    fig10, ax10 = validation_valSet_data.plot_time_series(fig10, ax10, 'competence', 0, "--b",
                                                          moving_average=10)
    fig10, ax10 = validation_valSet_data.plot_time_series(fig10, ax10, 'error', 0, "r", moving_average=10)

    plt.draw()
    plt.show()

show(block=True)
 
     
     
     
