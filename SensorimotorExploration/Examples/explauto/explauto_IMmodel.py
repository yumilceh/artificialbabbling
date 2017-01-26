'''
Created on Jan 23, 2017

@author: Juan Manuel Acevedo Valle
'''
from xmlrpclib import APPLICATION_ERROR
if __name__ == '__main__':
    import sys, os
    import numpy as np
    
    from Models.explauto_SM import explauto_SM
    from Models.explauto_IM import explauto_IM
    from DataManager.SimulationData import SimulationData
    from SensorimotorSystems.Parabola import ConstrainedParabolicArea as System
    # from SensorimotorSystems.Diva_Proprio2016a import Diva_Proprio2016a as System
    from Algorithm.CompetenceFunctions import get_competence_Baraglia2015 as get_competence
    from Algorithm.CompetenceFunctions import get_competence_Baraglia2015_explauto as competence
    from Algorithm.RndSensorimotorFunctions import get_random_motor_set, get_random_sensor_set
    from Algorithm.ModelEvaluation import SM_ModelEvaluation


    system = System()
    model_type = 'tree'  #'tree' 'discretized_progress''gmm_progress_beta'
    im_model = explauto_IM(system, model_type, competence)
    
    sm_model = explauto_SM(system, "nearest_neighbor")
    sigma_explo_ratio = 0.05
    sm_model.set_sigma_explo_ratio(sigma_explo_ratio)
    
    evaluation = SM_ModelEvaluation(system,
                                    0,
                                    sm_model)
    evaluation.loadEvaluationDataSet('../Parabola/parabola_validation_data_set_2.h5')
    
    simulation_data = SimulationData(system)
    
    for m in get_random_motor_set(system, 10):
        system.setMotorCommand(m)
        system.executeMotorCommand()        
        simulation_data.appendData(system)
        sm_model.train(simulation_data)
        
    for s_g in get_random_sensor_set(system, 10):
        system.sensor_goal = s_g
        sm_model.getMotorCommand(system)
        system.executeMotorCommand()        
        simulation_data.appendData(system)
        sm_model.train(simulation_data)
        im_model.train(simulation_data)
    
    n_experiments = 3000
    evaluation_samples = range(10,n_experiments+1, 1000)
    evaluation_error=[]
    
    for i in range(n_experiments):
        
        #---------------------------------------------------- #RANDOM EXPLORATION
        #-------------- system.sensor_goal = get_random_sensor_set(system, 1)[0]
        
        #------------------------------------------------------------- #CURIOSITY
        system.sensor_goal = im_model.get_interesting_goal(system)
        
        sm_model.getMotorCommand(system)
        system.executeMotorCommand() 
        get_competence(system)
        simulation_data.appendData(system)
        
        sm_model.train(simulation_data)
        im_model.train(simulation_data)
        if i in evaluation_samples:
            sm_model.set_sigma_explo_ratio(0)
            evaluation.model = sm_model
            eval_data = evaluation.evaluateModel()
            error_ = np.linalg.norm(eval_data.sensor_goal_data.data - eval_data.sensor_data.data, axis = 1)
            evaluation_error = np.append(evaluation_error, np.mean(error_))
            sm_model.set_sigma_explo_ratio(sigma_explo_ratio)
            
    
        #evaluate
    
    from DataManager.PlotTools import * 
    from matplotlib.pyplot import show
    
    fig1,ax1=initializeFigure()
    system.drawSystem(fig1,ax1)
    
    plt.plot(*s_g,  marker='o', color='blue')
    plt.hold(True)
    plt.plot(*system.sensorOutput,  marker='x', color='red')
        
    fig2,ax2=initializeFigure()
    simulation_data.plotSimulatedData2D(fig2,ax2,'sensor',0,'sensor',1,'or')
    simulation_data.plotSimulatedData2D(fig2,ax2,'sensor_goal',0,'sensor_goal',1,'xk')
    
     
    fig3,ax3=initializeFigure()
    simulation_data.plotTemporalSimulatedData(fig3,ax3,'competence',0,'b',moving_average=0)
    simulation_data.plotTemporalSimulatedData(fig3,ax3,'competence',0,'r',moving_average=10)
    t_evaluation = np.asmatrix(np.array(evaluation_samples)) + 20.
    plt.plot(t_evaluation.tolist()[0], evaluation_error, 'k')
    
    show(block=True)
 
     
     
     
