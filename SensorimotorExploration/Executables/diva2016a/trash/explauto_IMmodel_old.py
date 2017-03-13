"""
Created on Jan 23, 2017

@author: Juan Manuel Acevedo Valle
"""
if __name__ == '__main__':
    import numpy as np

    from SensorimotorExploration.Models.Interest.ExplautoIM import explauto_IM
    from SensorimotorExploration.DataManager.SimulationData import SimulationData

    from SensorimotorExploration.Systems.Diva2016a import Diva2016a as System

    # from Algorithm.CompetenceFunctions import comp_Baraglia2015 as get_competence
    # from Algorithm.CompetenceFunctions import comp_Baraglia2015_expl as competence

    from SensorimotorExploration.Algorithm.utils.competence_funcs import comp_Moulin2013 as get_competence
    from SensorimotorExploration.Algorithm.utils.competence_funcs import \
        comp_Moulin2013_expl as competence

    from SensorimotorExploration.Algorithm.utils.functions import get_random_motor_set, get_random_sensor_set
    from SensorimotorExploration.Algorithm.ModelEvaluation import SM_ModelEvaluation

    system = System()
    model_type = 'tree'  # 'tree' 'discretized_progress''gmm_progress_beta'
    im_model = explauto_IM(system, model_type, competence)

    sm_model = Explauto_SM(system, "nearest_neighbor")
    sigma_explo_ratio = 0.1
    sm_model.set_sigma_explo_ratio(sigma_explo_ratio)

    evaluation = SM_ModelEvaluation(system,
                                    400,
                                    sm_model)
    evaluation.setValidationEvaluationSets()
    simulation_data = SimulationData(system)

    init_samples = 100

    for m in get_random_motor_set(system, init_samples):
        system.set_action(m)
        system.executeMotorCommand()
        simulation_data.appendData(system)
        sm_model.train(simulation_data)

    for s_g in get_random_sensor_set(system, init_samples):
        system.sensor_goal = s_g
        sm_model.get_action(system)
        system.executeMotorCommand()
        simulation_data.appendData(system)
        sm_model.train(simulation_data)
        im_model.train(simulation_data)

    n_experiments = 15000
    evaluation_samples = range(10, n_experiments + 1, 3000)
    evaluation_error = []
    mean_competence = []

    for i in range(n_experiments):

        # --------------------------------------------------- #RANDOM EXPLORATION
        # -------------- system.sensor_goal = get_random_sensor_set(system, 1)[0]
        # ------------------------------------------------------------ #CURIOSITY
        system.sensor_goal = im_model.get_goal(system)

        sm_model.get_action(system)
        system.executeMotorCommand()
        get_competence(system)
        simulation_data.appendData(system)

        sm_model.train(simulation_data)
        im_model.train(simulation_data)
        if i in evaluation_samples:
            sm_model.set_sigma_explo_ratio(0)
            evaluation.model = sm_model
            eval_data = evaluation.evaluateModel()
            error_ = np.linalg.norm(eval_data.sensor_goal_data.data - eval_data.sensor_data.data, axis=1)
            evaluation_error = np.append(evaluation_error, np.mean(error_))
            mean_competence = np.append(mean_competence, np.mean(eval_data.competence_data.datal))
            sm_model.set_sigma_explo_ratio(sigma_explo_ratio)


            # evaluate

    from SensorimotorExploration.DataManager.PlotTools import *
    from matplotlib.pyplot import show

    fig2, ax2 = initializeFigure()
    simulation_data.plotSimulatedData2D(fig2, ax2, 'sensor', 0, 'sensor', 1, 'or')
    simulation_data.plotSimulatedData2D(fig2, ax2, 'sensor_goal', 0, 'sensor_goal', 1, 'xk')

    fig3, ax3 = initializeFigure()
    simulation_data.plotTemporalSimulatedData(fig3, ax3, 'competence', 0, 'b', moving_average=0)
    simulation_data.plotTemporalSimulatedData(fig3, ax3, 'competence', 0, 'r', moving_average=10)
    t_evaluation = np.asmatrix(np.array(evaluation_samples)) + 20.
    plt.plot(t_evaluation.tolist()[0], evaluation_error, 'k')

    show(block=True)
