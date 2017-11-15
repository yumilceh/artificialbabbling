"""
Created on Mar 17, 2017

@author: Juan Manuel Acevedo Valle
"""
import datetime
import itertools
import os
import time
from multiprocessing import Process

from exploration.algorithm.evaluation import Evaluation
from exploration.algorithm.algorithm_2017 import Algorithm as Algorithm
from exploration.algorithm.trash.algorithm_2015 import OBJECT
#  Adding libraries##
from exploration.systems.Diva2017a import Diva2017a as System
from exploration.systems.Diva2017a import Instructor
from diva_configurations import model_, comp_func

directory = 'diva_IEEE_SI/experiment_IEEE_SI_slopes_cmf_050'

# models
f_sm_key = 'igmm_sm'
f_cons_key = 'explauto_cons'
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

    n_initialization = 1000
    n_experiments = 100000
    n_save_data = 10000  # np.nan to not save, -1 to save 5 times during exploration
    eval_step = 2500 #np.nan to not evaluate

    #random_seeds, mode_ops, social_slopes, vowel_units
    random_seed = ops[0]
    mode_ = ops[1]
    social_slope_ = ops[2]
    vowel_units_ = ops[3]

    system = System()
    instructor = Instructor(n_su = vowel_units_, slope=social_slope_)

    # Creating models ##
    models = OBJECT()
    models.f_sm = model_(f_sm_key, system)
    models.f_cons = model_(f_cons_key, system)
    models.f_im = model_(f_im_key, system)

    # Creating Simulation object, running simulation and plotting experiments##
    # tree/DP Interest Model
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")

    file_prefix = directory + '/IEEE_TCDS_SI_' + str(idx) + '_'+ now

    evaluation_sim = Evaluation(system,
                                models.f_sm, comp_func=comp_func,
                                file_prefix=file_prefix)

    evaluation_sim.data = instructor.data
    evaluation_sim.n_samples_val = len(evaluation_sim.data.sensor.data)
    evaluation_sim.random_indexes_val = range(evaluation_sim.n_samples_val)

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
    simulation.f_cons_key = f_cons_key
    simulation.f_im_key = f_im_key

    simulation.mode = mode_

    simulation.run(proprio=True)
    simulation.do_evaluation(0, force=True, save_data=True)

    try:
        import numpy as np
        np.savetxt(file_prefix + '_instructor_thresh.txt', instructor.unit_threshold)
    except:
        pass

if __name__ == '__main__':
    try:
        os.mkdir(directory)
    except OSError:
        print('WARNING. Directory already exists.')

    # 2469, 147831, 1234
    random_seeds = [1321,1457, 283, 2469, 147831, 1234]
    mode_ops = ['social']
    social_slopes = [0.93]# [1., 0.999999, 0.99, 0.96, 0.93]
    vowel_units = [50]#323,223,123,50]

    groups1 = itertools.product(random_seeds, mode_ops, social_slopes, vowel_units)
    groups2 = itertools.product(random_seeds, ['autonomous'], [0.999999],vowel_units)

    processes = []
    max_processes = 3

    for idx, ops in enumerate(list(groups1)):#+list(groups2)):
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
