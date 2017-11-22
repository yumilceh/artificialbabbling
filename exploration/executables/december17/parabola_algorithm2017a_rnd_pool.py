"""
Created on Nov 21, 2017

@author: Juan Manuel Acevedo Valle
"""
import datetime
import os
import time
import copy

from multiprocessing import Process
from exploration.algorithm.algorithm2017 import Algorithm as Algorithm
from exploration.algorithm.evaluation import Evaluation
from exploration.algorithm.algorithm2017 import OBJECT


def sim_agent(ops):
    n_initialization = ops['n_initialization']
    n_experiments = ops['n_experiments']
    n_save_data = ops['n_save_data']
    eval_step = ops['eval_step']
    random_seed = ops['random_seed']
    mode_ = ops['mode']
    system = ops['system']
    instructor = ops['instructor']
    models = ops['models']
    proprio = ops['Proprio']

    directory = ops['directory_results']
    file_prefix = ops['file_prefix']
    evaluation_sim = ops['evaluation']
    comp_func = ops['comp_func']

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")

    file_prefix = directory + '/' + file_prefix  + '_'+ now

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

    simulation.mode = mode_

    simulation.run(proprio=proprio)
    simulation.do_evaluation(0, force=True, save_data=True)

    try:
        import numpy as np
        np.savetxt(file_prefix + '_instructor_thresh.txt', instructor.unit_threshold)
    except:
        pass


def sim_pool(*args, **ops):
    try:
        os.mkdir(ops['directory_results'])
    except OSError:
        print('WARNING. Directory already exists.')

    # 2469, 147831, 1234

    processes = []
    max_processes = 3

    system = []
    instructor = []
    models = []
    evaluation = []

    for random_seed in ops['random_seeds']:
        time.sleep(5)
        system += [copy.deepcopy(args[0])]
        instructor += [copy.deepcopy(args[1])]
        models += [copy.deepcopy(args[2])]
        evaluation += [copy.deepcopy(args[3])]

        ops_ = ops.copy()
        ops_['random_seed'] = random_seed
        ops_['system'] = system[-1]
        ops_['instructor'] = instructor[-1]
        ops_['models'] = models[-1]
        ops_['evaluation'] = evaluation[-1]

        # Creating Agent ##
        processes += [Process(target=sim_agent, args=(ops_,))]
        # processes[-1].daemon = True
        processes[-1].start()
        # processes[-1].join()
        while len(processes) >= max_processes:
            time.sleep(5)
            for i, t in enumerate(processes):
                if not t.is_alive():
                    del system[i]
                    del instructor[i]
                    del models[i]
                    del evaluation[i]
                    del processes[i]  # pop

    while len(processes) > 0:
        time.sleep(5)
        for i, t in enumerate(processes):
            if not t.is_alive():
                del system[i]
                del instructor[i]
                del models[i]
                del evaluation[i]
                del processes[i]

    print('The work  is done!!')

if __name__ == '__main__':
    from exploration.systems.parabola import ParabolicRegion
    from exploration.systems.parabola import Instructor
    from parabola_configurations import model_, comp_func

    ops = {}
    directory = 'test/parabola_IMmodel'
    ops['random_seeds'] = [1321,1457, 283, 2469, 147831, 1234]
    ops['directory_results'] = directory
    ops['file_prefix'] = 'tune IM model'
    ops['n_initialization'] = 100
    ops['n_experiments'] = 10000
    ops['n_save_data'] = 5000
    ops['eval_step'] = 500

    ops['mode'] = 'social'
    ops['Proprio'] = True
    ops['comp_func'] = comp_func

    system = ParabolicRegion()
    
    f_im_key, f_sm_key, f_cons_key = 'explauto_im', 'igmm_sm', 'explauto_cons'
    models = OBJECT()
    models.f_sm = model_(f_sm_key, system)
    models.f_im = model_(f_im_key, system)
    models.f_cons = model_(f_cons_key, system)
    
    val1_name, val1_file = 'whole', '../../systems/datasets/parabola_v2_dataset.h5'.replace('/', os.sep)
    val2_name, val2_file = 'social', '../../systems/datasets/instructor_parabola_1.h5'.replace('/', os.sep)
    evaluation =  Evaluation(system,
                             models.f_sm,
                             comp_func=comp_func,
                             file_prefix=ops['file_prefix'])

    evaluation.load_eval_dataset(val1_file, name = val1_name)
    evaluation.load_eval_dataset(val2_file, name = val2_name)

    args = [system, Instructor(), models, evaluation]

    sim_pool(*args, **ops)
