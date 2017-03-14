"""
Created on Mar 13, 2017

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
    from SensorimotorExploration.Systems.Diva2016a import Diva2016a as System
    from SensorimotorExploration.Systems.Diva2016a import Instructor
    from SensorimotorExploration.Algorithm.algorithm_2017a import InteractionAlgorithm as Algorithm
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

    n_initialization = 100
    n_experiments = 20000
    n_save_data = 5000   # np.nan to not save, -1 to save 5 times during exploration

    eval_step = 5000

    random.seed(random_seed)
    np_rnd.seed(random_seed)

    # Creating Agent ##
    system = System()
    instructor = Instructor()

    # Creating Models ##
    models = OBJECT()

    models.f_sm = model_(f_sm_key, system)
    models.f_ss = model_(f_ss_key, system)
    models.f_im = model_(f_im_key, system)

    # Creating Simulation object, running simulation and plotting experiments##
    # tree/DP Interest Model
    file_prefix = 'Vowels_Tree_' + now

    evaluation_sim = SM_ModelEvaluation(system,
                                        models.f_sm, comp_func=comp_func,
                                        file_prefix=file_prefix)

    evaluation_sim.loadEvaluationDataSet('../../Systems/datasets/vowels_dataset_1.h5')

    proprio = True


    simulation = Algorithm(system, instructor,
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

    simulation.f_sm_key = f_sm_key
    simulation.f_ss_key = f_ss_key
    simulation.f_im_key = f_im_key

    simulation.run(proprio=proprio)

    sim_data = simulation.data

    evaluation_sim.model.set_sigma_explo_ratio(0.)
    val_data = evaluation_sim.evaluateModel(saveData=True)
    error_ = np.linalg.norm(val_data.sensor_goal_data.data.as_matrix() -
                            val_data.sensor_data.data.as_matrix(), axis=1)
    print("Mean evaluation error is {} (max: {}, min: {})".format(np.mean(error_),
                                                                  np.max(error_),
                                                                  np.min(error_)))

    from plot_results import show_results

    show_results(system, simulation, val_data)
