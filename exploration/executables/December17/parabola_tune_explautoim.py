"""
Created on Nov 22, 2017

@author: Juan Manuel Acevedo Valle
"""
import datetime, time, copy, random, pickle, os, hyperopt

from multiprocessing import Process
from exploration.algorithm.algorithm2017 import Algorithm as Algorithm
from exploration.algorithm.evaluation import Evaluation
from exploration.algorithm.algorithm2017 import OBJECT

import numpy as np
from hyperopt import Trials, STATUS_OK
from hyperopt.hp import uniform, choice


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

    file_prefix = directory + '/' + file_prefix + '_' + now

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
    max_processes = 2

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
    from parabola_configurations import model_, comp_func, comp_func_expl
    from exploration.models.Interest.ExplautoIM import explauto_IM

    FLAGS = None

    results_folder = 'test/parabola_IMmodel'.replace('/', os.sep)
    try:
        os.mkdir(results_folder)
    except OSError:
        print('WARNING. Directory already exists.')

    load_trials = True
    save_trials = True
    if load_trials:
        try:
            trials = pickle.load(open(results_folder + 'trials.hyperopt', "rb"))
        except:
            trials = Trials()
            print('no trials found')
    else:
        trials = Trials()

    model_specs = {'model_type': 'discretized_progress'}

    experiment_specs = {}
    space = {'x_card': choice('x_card', np.arange(500, 1500)),
             'win_size': choice('win_size', np.arange(1, 20)),
             'eps_random': uniform('eps_random', 0.01, .99)}

    # Cross validation
    max_evals = 150
    k_folds = 100

    def objective(space_):
        print('\n')
        print('\n')
        print('Iteration: ' + str(len(trials.losses())))
        print('\n')

        # Specs
        specs = space_.copy()
        specs.update(model_specs)

        ops = {}
        directory = 'test/parabola_IMmodel'
        ops['random_seeds'] =  range(k_folds) # [1321, 1457, 283, 2469, 147831, 1234]
        ops['directory_results'] = directory
        ops['file_prefix'] = 'tune IM model_' + str(len(trials.losses()))
        ops['n_initialization'] = 100
        ops['n_experiments'] = 10000
        ops['n_save_data'] = 5000
        ops['eval_step'] = 500

        ops['mode'] = 'social'
        ops['Proprio'] = True
        ops['comp_func'] = comp_func
        ops['comp_func_expl'] = comp_func_expl

        f_sm_key, f_cons_key = 'igmm_sm', 'explauto_cons'
        models = OBJECT()
        models.f_sm = model_(f_sm_key, ParabolicRegion())
        models.f_cons = model_(f_cons_key, ParabolicRegion())

        model_conf = {specs['model_type']: {'x_card': specs['x_card'],
                                            'win_size': specs['win_size'],
                                            'eps_random': specs['eps_random']}}  # 'discretized_progress'

        f_im = explauto_IM(ParabolicRegion(),  ops['comp_func_expl'], specs['model_type'], **{'model_conf': model_conf})

        models.f_im = f_im

        val1_name, val1_file = 'whole', '../../systems/datasets/parabola_v2_dataset.h5'.replace('/', os.sep)
        val2_name, val2_file = 'social', '../../systems/datasets/instructor_parabola_1.h5'.replace('/', os.sep)

        evaluation = Evaluation(ParabolicRegion(),
                                models.f_sm,
                                comp_func=comp_func,
                                file_prefix=ops['file_prefix'])

        evaluation.load_eval_dataset(val1_file, name=val1_name)
        evaluation.load_eval_dataset(val2_file, name=val2_name)

        args = [ParabolicRegion(), Instructor(), models, evaluation]

        sim_pool(*args, **ops)

        social_eva_errors = []
        whole_eva_errors = []
        data_files = os.listdir(directory)
        for data_file in (d_f for d_f in data_files if (ops['file_prefix'] in d_f and '_social_eval_error.txt' in d_f)):
            with open(directory + '/' + data_file, 'r') as f:
                for line in f:
                    line.replace('\n', '')
                    eva_errors_str = line.split(': ')
                social_eva_errors += [float(eva_errors_str[1])]
            with open(directory + '/' + data_file.replace('social','whole'), 'r') as f:
                for line in f:
                    line.replace('\n', '')
                    eva_errors_str = line.split(': ')
                whole_eva_errors += [float(eva_errors_str[1])]

        loss_av = 0.6*np.mean(social_eva_errors)+0.4*np.mean(whole_eva_errors[-1])

        if len(trials.losses()) == 1:
            best_iteration = len(trials.losses())
        elif loss_av < np.min(trials.losses()[:-1]):
            best_iteration = len(trials.losses())
        else:
            best_iteration = np.argmin(trials.losses()[:-1]) + 1

        # Best iteration
        print("Best iterarion is: {}".format(best_iteration))
        print("This iteration model got val_loss={}, social_error={} and whole_error={} with:".format(loss_av,
                                                                                                      social_eva_errors[-1],
                                                                                                      whole_eva_errors[-1]))
        for key in specs.keys():
            print("{}: {}".format(key, specs[key]))

        # Save trials
        if save_trials:
            with open(results_folder + 'trials.hyperopt', 'wb') as f:
                pickle.dump(trials, f)
            f.close()

        return {'loss': loss_av, 'status': STATUS_OK}

    def optimize():
        best_param = hyperopt.fmin(
            objective,
            space=space,
            algo=hyperopt.tpe.suggest,
            max_evals=max_evals + len(trials.trials),
            trials=trials,
            verbose=1)
        lowest_loss_ind = np.argmin(trials.losses())

        print('\n')
        print('\n')
        print('best iteration: ' + str(lowest_loss_ind + 1))
        print('best hyperparameters: ' + str(best_param))
        print('number of iterations: ' + str(len(trials.trials)))

        with open('best_params.pickle', 'wb') as handle:
            pickle.dump(best_param, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return best_param




    best_param = optimize()

        ###################################33



    # 'tree':                 {'max_points_per_region': 100,
    #                          'max_depth': 20,
    #                          'split_mode': 'best_interest_diff',
    #                          'progress_win_size': 50,
    #                          'progress_measure': 'abs_deriv_smooth',
    #                          'sampling_mode': {'mode': 'softmax',
    #                                            'param': 0.2,
    #                                            'multiscale': False,
    #                                            'volume': True}},


