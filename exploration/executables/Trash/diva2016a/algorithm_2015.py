"""
Created on Mar 8, 2017

@author: Juan Manuel Acevedo Valle
"""
import datetime
import random

from numpy import random as np_rnd

now = datetime.datetime.now().strftime("alg2015_%Y_%m_%d_%H_%M_")

if __name__ == '__main__':
    #  Adding libraries##
    from exploration.systems.Diva2016a import Diva2016a as System
    from exploration.algorithm.trash.algorithm_2015 import Algorithm_2015 as Algorithm
    from exploration.algorithm.trash.algorithm_2015 import OBJECT
    from exploration.algorithm.trash.ModelEvaluation import SM_ModelEvaluation
    from exploration.data.PlotTools import *

    from model_configurations import model_, comp_func


    # models
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

    # Creating models ##
    models = OBJECT()

    models.f_sm = model_(f_sm_key, system)
    models.f_ss = model_(f_ss_key, system)
    models.f_im = model_(f_im_key, system)

    evaluation_sim = SM_ModelEvaluation(system,
                                        models.f_sm, comp_func=comp_func)

    evaluation_sim.loadEvaluationDataSet('../../systems/datasets/vowels_dataset_1.h5')

    #  Creating Simulation object, running simulation and plotting experiments##
    proprio = True
    #(P/NP) Proprio or not, expla/igmm SM Model, tree/DP Interest Model
    file_prefix = 'Eva_vowels_P_igmm_Tree_' + now

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
                           sm_all_samples=False,
                           file_prefix=file_prefix)

    if proprio:
        simulation.run_proprio()
    else:
        simulation.run_simple()

    sim_data = simulation.data

    evaluation_sim.model.set_sigma_explo_ratio(0.)
    val_data = evaluation_sim.evaluate_model(saveData=False)
    error_ = np.linalg.norm(val_data.sensor_goal_data.data.as_matrix() -
                            val_data.sensor_data.data.as_matrix(), axis=1)
    print("Mean evaluation error is {} (max: {}, min: {})".format(np.mean(error_),
                                                                  np.max(error_),
                                                                  np.min(error_)))

    from plot_results import show_results

    show_results(system, simulation, val_data)
