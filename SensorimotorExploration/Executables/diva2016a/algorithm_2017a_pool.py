"""
Created on Mar 17, 2017

@author: Juan Manuel Acevedo Valle
"""
from multiprocessing import Process
import time
import itertools
from numpy import random as np_rnd
import datetime
import os, sys, random


#  Adding libraries##
from SensorimotorExploration.Systems.Diva2016a import Diva2016a as System
from SensorimotorExploration.Systems.Diva2016a import Instructor
from SensorimotorExploration.Algorithm.algorithm_2017a import InteractionAlgorithm as Algorithm
from SensorimotorExploration.Algorithm.algorithm_2015 import OBJECT
from SensorimotorExploration.Algorithm.ModelEvaluation import SM_ModelEvaluation
from SensorimotorExploration.DataManager.PlotTools import *

from model_configurations import model_, comp_func

directory = 'experiment_1'
#

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

def sim_agent(ops):
    # print('new_thread')
    # for i in range(10):
    #     print(i)
    #     time.sleep(1)
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

    # Creating Simulation object, running simulation and plotting experiments##
    # tree/DP Interest Model
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_")

    file_prefix = directory + '/Vowels_Tree_' + now

    evaluation_sim = SM_ModelEvaluation(system,
                                        models.f_sm, comp_func=comp_func,
                                        file_prefix=file_prefix)

    evaluation_sim.loadEvaluationDataSet('../../Systems/datasets/vowels_dataset_1.h5')

    simulation = Algorithm(system,
                          models,
                          n_experiments,
                          comp_func,
                          instructor=instructor,
                          n_initialization_experiments=n_initialization,
                          random_seed=random_seed,
                          g_im_initialization_method='all',
                          n_save_data=n_save_data,
                          evaluation=evaluation_sim,
                          eval_step=eval_step,
                          sm_all_samples=False,
                          file_prefix=file_prefix)

    simulation.f_sm_key = f_sm_key
    simulation.f_ss_key = f_ss_key
    simulation.f_im_key = f_im_key

    simulation.mode = mode_

    simulation.run(proprio=proprio)
    simulation.do_evaluation(0, force=True, save_data=True)

if __name__ == '__main__':
    try:
        os.mkdir(directory)
    except OSError:
        print('WARNING. Directory already exists.')

    n_initialization = 100
    n_experiments = 20000
    n_save_data = 5000  # np.nan to not save, -1 to save 5 times during exploration

    eval_step = np.nan

    random_seeds = [1234, 1321, 1457, 283, 2469, 147831, 234096, 2453, 2340554, 12455] # 8975, 91324, 752324, 1264183,

    proprio_ops = [True, False]
    mode_ops = ['autonomous', 'social']

    threads = []
    max_threads = 4

    for idx, ops in enumerate(itertools.product(random_seeds, proprio_ops, mode_ops)):
        # Creating Agent ##

        threads += [Process(target=sim_agent, args=(ops,))]
        # threads[-1].daemon = True
        threads[-1].start()
        # threads[-1].join()
        time.sleep(5)
        while len(threads) >= max_threads:
            for i, t in enumerate(threads):
                if not t.is_alive():
                    del threads[i]  # pop


