"""
Created on Mar 13, 2017

@author: Juan Manuel Acevedo Valle
"""
import datetime

from numpy import random as np_rnd

now = datetime.datetime.now().strftime("Social_%Y_%m_%d_%H_%M_")

if __name__ == '__main__':
    #  Adding the projects folder to the path##
    import random

    # sys.path.append("../../")

    #  Adding libraries##
    from exploration.systems.Diva2016a import Diva2017a as System
    from exploration.systems.Diva2016a import Instructor
    from exploration.algorithm.algorithm2017 import Algorithm as Algorithm
    from exploration.algorithm.trash.algorithm_2015 import OBJECT
    from exploration.algorithm.trash.ModelEvaluation import SM_ModelEvaluation
    from exploration.data.PlotTools import *

    from diva_configurations import model_, comp_func

    # models
    f_sm_key = 'igmm_sm'
    f_ss_key = 'explauto_cons'
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

    # Creating models ##
    models = OBJECT()

    models.f_sm = model_(f_sm_key, system)
    models.f_ss = model_(f_ss_key, system)
    models.f_im = model_(f_im_key, system)

    # Creating Simulation object, running simulation and plotting experiments##
    # tree/DP Interest Model
    directory = 'RndExperiments'
    file_prefix = directory + '/Vowels_Tree_' + now

    evaluation_sim = SM_ModelEvaluation(system,
                                        models.f_sm, comp_func=comp_func,
                                        file_prefix=file_prefix)

    evaluation_sim.loadEvaluationDataSet('../../systems/datasets/vowels_dataset_1.h5')

    proprio = True
    mode = 'social'

    simulation = Algorithm(system,
                           models,
                           n_experiments,
                           comp_func,
                           instructor = instructor,
                           n_initialization_experiments=n_initialization,
                           random_seed=1234,
                           g_im_initialization_method='all',
                           n_save_data=n_save_data,
                           evaluation=evaluation_sim,
                           eval_step=eval_step,
                           sm_all_samples=False,
                           file_prefix=file_prefix)

    simulation.f_sm_key = f_sm_key
    simulation.f_ss_key = f_ss_key
    simulation.f_im_key = f_im_key

    simulation.mode = mode

    simulation.run(proprio=proprio)

    sim_data = simulation.data

    evaluation_sim.model.set_sigma_explo_ratio(0.)
    val_data = evaluation_sim.evaluate_model(saveData=True)
    error_ = np.linalg.norm(val_data.sensor_goal.data.as_matrix() -
                            val_data.sensor.data.as_matrix(), axis=1)
    print("Mean evaluation error is {} (max: {}, min: {})".format(np.mean(error_),
                                                                  np.max(error_),
                                                                  np.min(error_)))
