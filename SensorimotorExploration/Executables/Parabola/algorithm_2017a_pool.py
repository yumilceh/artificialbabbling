"""
Created on Mar 8, 2017

@author: Juan Manuel Acevedo Valle
"""
from numpy import linspace
from numpy import random as np_rnd
import itertools
import datetime

if __name__ == '__main__':
    #  Adding the projects folder to the path##
    import os, sys, random

    # sys.path.append("../../")

    #  Adding libraries##
    from SensorimotorExploration.Systems.Parabola import ParabolicRegion as System
    from SensorimotorExploration.Systems.Parabola import Instructor
    from SensorimotorExploration.Algorithm.algorithm_2017a import InteractionAlgorithm as Algorithm
    from SensorimotorExploration.Algorithm.algorithm_2015 import OBJECT
    from SensorimotorExploration.Algorithm.ModelEvaluation import SM_ModelEvaluation


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
    n_initialization = 100
    n_experiments = 2000
    n_save_data = 2000   # np.nan to not save, -1 to save 5 times during exploration

    eval_step = 400

    # random.seed(random_seed)
    # np_rnd.seed(random_seed)
    directory = 'experiment_4'
    os.mkdir(directory)


    random_seeds = [1234, 1321, 1457, 283, 2469, 147831, 234096, 2453, 2340554, 12455, 8975, 91324,752324,1264183, 82376, 92835, 823975, 2376324]
    proprio_ops = [True, False]
    mode_ops = ['autonomous','social']

    for idx,ops in enumerate(itertools.product(random_seeds, proprio_ops, mode_ops)):
        # Creating Agent ##
        system = System()
        instructor = Instructor()

        random_seed = ops[0]
        proprio = ops[1]
        mode_ = ops[2]

        # Creating Models ##
        models = OBJECT()

        models.f_sm = model_(f_sm_key, system)
        models.f_ss = model_(f_ss_key, system)
        models.f_im = model_(f_im_key, system)

        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_")
        file_prefix = directory + '/Parabola_Pool_' + str(idx) + '_' + now

        #  Creating Simulation object, running simulation and plotting experiments##
        evaluation_sim = SM_ModelEvaluation(system,
                                            models.f_sm, comp_func=comp_func)

        evaluation_sim.loadEvaluationDataSet('../../Systems/datasets/parabola_dataset_1.h5')

        simulation = Algorithm(system,
                               models,
                               n_experiments,
                               comp_func,
                               instructor = instructor,
                               n_initialization_experiments=n_initialization,
                               random_seed=random_seed,
                               evaluation=evaluation_sim,
                               eval_step=eval_step,
                               g_im_initialization_method='all', #'all'
                               n_save_data=n_save_data,
                               sm_all_samples=False,
                               file_prefix=file_prefix)

        simulation.mode = mode_ # social or autonomous

        simulation.f_sm_key = f_sm_key
        simulation.f_ss_key = f_ss_key
        simulation.f_im_key = f_im_key

        simulation.run(proprio=proprio)

        evaluation_sim = SM_ModelEvaluation(system,
                                            simulation.models.f_sm,
                                            comp_func=comp_func,
                                            file_prefix=file_prefix + 'whole_')

        evaluation_sim.model.set_sigma_explo_ratio(0.)
        evaluation_sim.model.mode = 'exploit'

        evaluation_sim.loadEvaluationDataSet('../../Systems/datasets/parabola_dataset_1.h5')
        evaluation_sim.model.set_sigma_explo_ratio(0.)
        val_data = evaluation_sim.evaluateModel(saveData=True)
        del(evaluation_sim)

        evaluation_sim = SM_ModelEvaluation(system,
                                            simulation.models.f_sm,
                                            comp_func=comp_func,
                                            file_prefix=file_prefix + 'social_')
        evaluation_sim.loadEvaluationDataSet('../../Systems/datasets/instructor_parabola_1.h5')
        val_data = evaluation_sim.evaluateModel(saveData=True)

        del simulation
        del models
        del evaluation_sim
