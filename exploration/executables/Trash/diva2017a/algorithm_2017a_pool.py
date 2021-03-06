"""
Created on Mar 17, 2017

@author: Juan Manuel Acevedo Valle
"""
import datetime
import itertools
import os
import time
from multiprocessing import Process

from exploration.algorithm.algorithm2017 import Algorithm as Algorithm
from exploration.algorithm.trash.ModelEvaluation import SM_ModelEvaluation
from exploration.algorithm.trash.algorithm_2015 import OBJECT
#  Adding libraries##
from exploration.systems.Diva2017a import Diva2017a as System
from exploration.systems.Diva2017a import Instructor
from model_configurations import model_, comp_func

directory = 'experiment_5_vowels'

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

def sim_agent(ops,idx):
    system = System()
    instructor = Instructor()

    random_seed = ops[0]
    proprio = ops[1]
    mode_ = ops[2]

    # Creating models ##
    models = OBJECT()
    models.f_sm = model_(f_sm_key, system)
    models.f_ss = model_(f_ss_key, system)
    models.f_im = model_(f_im_key, system)

    # Creating Simulation object, running simulation and plotting experiments##
    # tree/DP Interest Model
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_")

    file_prefix = directory + '/Vowels_Tree_' + str(idx) + '_'+ now

    evaluation_sim = SM_ModelEvaluation(system,
                                        models.f_sm, comp_func=comp_func,
                                        file_prefix=file_prefix)

    evaluation_sim.loadEvaluationDataSet('../../systems/datasets/german_dataset_3.h5')

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
    n_experiments = 250000
    n_save_data = 10000  # np.nan to not save, -1 to save 5 times during exploration

    eval_step = 5000 #np.nan to not evaluate

    # ,
    random_seeds = [1234,1321,1457, 283,2469, 147831]
    proprio_ops = [True, False]
    mode_ops = ['social', 'autonomous']

    processes = []
    max_processes = 4

    for idx, ops in enumerate(itertools.product(random_seeds, proprio_ops, mode_ops)):
        idx2 = idx
        # Creating Agent ##

        processes += [Process(target=sim_agent, args=(ops,idx2))]
        # processes[-1].daemon = True
        processes[-1].start()
        # processes[-1].join()
        while len(processes) >= max_processes:
            time.sleep(5)
            for i, t in enumerate(processes):
                if not t.is_alive():
                    del processes[i]  # pop

    while len(processes) > 0:
        time.sleep(5)
        for i, t in enumerate(processes):
            if not t.is_alive():
                del processes[i]

    print('The work  is done!!')