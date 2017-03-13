"""
Created on Mar 8, 2017

@author: Juan Manuel Acevedo Valle
"""
from numpy import linspace
from numpy import random as np_rnd
import datetime

now = datetime.datetime.now().strftime("Social_%Y_%m_%d_%H_%M_")

if __name__ == '__main__':
    #  Adding the projects folder to the path##
    import os, sys, random

    # sys.path.append("../../")

    #  Adding libraries##
    from SensorimotorExploration.Systems.Parabola import ParabolicRegion as System
    from SensorimotorExploration.Algorithm.algorithm_2015 import Algorithm_2015 as Algorithm
    from SensorimotorExploration.Algorithm.algorithm_2015 import OBJECT
    from SensorimotorExploration.Algorithm.ModelEvaluation import SM_ModelEvaluation
    from SensorimotorExploration.DataManager.PlotTools import *
    from SensorimotorExploration.Algorithm.utils.functions import generate_motor_grid

    from model_configurations import model_, comp_func

    # Models
    f_sm_key = 'igmm_sm'
    f_ss_key = 'explauto_ss'
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

    # To guarantee reproducible experiments
    random_seed = 1234

    n_initialization = 40
    n_experiments = 1000
    n_save_data = np.nan   # np.nan to not save, -1 to save 5 times during exploration

    eval_step = 100

    random.seed(random_seed)
    np_rnd.seed(random_seed)

    # Creating Agent ##
    system = System()

    # Creating Models ##
    models = OBJECT()

    models.f_sm = model_(f_sm_key, system)
    models.f_ss = model_(f_ss_key, system)
    models.f_im = model_(f_im_key, system)

    evaluation_sim = SM_ModelEvaluation(system,
                                        models.f_sm, comp_func=comp_func)

    evaluation_sim.loadEvaluationDataSet('../../Systems/datasets/parabola_validation_data_set_2.h5')

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
                           eval_step=eval_step,
                           sm_all_samples=False)

    simulation.run_proprio()

    sim_data = simulation.data

    evaluation_sim.model.set_sigma_explo_ratio(0.)
    val_data = evaluation_sim.evaluateModel(saveData=False)
    error_ = np.linalg.norm(val_data.sensor_goal_data.data.as_matrix() -
                            val_data.sensor_data.data.as_matrix(), axis=1)
    print("Mean evaluation error is {} (max: {}, min: {})".format(np.mean(error_),
                                                                  np.max(error_),
                                                                  np.min(error_)))

    #  Looking at the proprioceptive model
    somato_th = system.somato_threshold

    n_motor_samples = 1000

    m1, m2 = generate_motor_grid(system, n_motor_samples)

    proprio_val = []
    for m in zip(m1.flatten(), m2.flatten()):
        system.somato_out = 0.
        system.set_action(np.array([m[0], m[1]]))
        somato_pred = simulation.models.f_ss.predict_somato(system)
        system.executeMotorCommand()
        somato_res = system.somato_out
        # print("We predicted {} but got {}.".format(somato_pred, somato_res))
        system.executeMotorCommand_unconstrained()

        if somato_pred >= somato_th and somato_res >= somato_th:
            proprio_val += [[system.sensor_out[0], system.sensor_out[1], '.k']]

        if somato_pred >= somato_th > somato_res:
            proprio_val += [[system.sensor_out[0], system.sensor_out[1], 'xr']]

        if somato_pred < somato_th and somato_res < somato_th:
            proprio_val += [[system.sensor_out[0], system.sensor_out[1], '.b']]

        if somato_pred < somato_th <= somato_res:
            proprio_val += [[system.sensor_out[0], system.sensor_out[1], 'xk']]

    from plot_results import show_results

    show_results(system, simulation, val_data, proprio_val)
