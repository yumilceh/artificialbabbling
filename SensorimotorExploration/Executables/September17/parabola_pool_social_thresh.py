"""
Created on Mar 8, 2017

@author: Juan Manuel Acevedo Valle
"""
import datetime
import itertools

if __name__ == '__main__':

    #  Adding the projects folder to the path##
    import os

    # sys.path.append("../../")

    #  Adding libraries##
    from SensorimotorExploration.Systems.Parabola_v2 import ParabolicRegion as System
    from SensorimotorExploration.Systems.Parabola_v2 import Instructor
    from SensorimotorExploration.Algorithm.algorithm_2017a import InteractionAlgorithm as Algorithm
    from SensorimotorExploration.Algorithm.trash.algorithm_2015 import OBJECT
    from SensorimotorExploration.Algorithm.ModelEvaluation_v2 import SM_ModelEvaluation
    from parabola_configurations import model_, comp_func

    # Models
    f_sm_key = 'igmm_sm'
    f_ss_key = 'igmm_ss'
    f_cons_key = 'explauto_cons'

    """
       'igmm_sm': IGMM_SM,
       'igmm_ss': IGMM_SS,
       'explauto_im': ea_IM,
       'explauto_cons': ea_cons
    """

    # To guarantee reproducible experiments
    n_initialization = 100
    n_experiments = 25000
    n_save_data = 2000   # np.nan to not save, -1 to save 5 times during exploration

    eval_step = 500

    # random.seed(random_seed)
    # np_rnd.seed(random_seed)
    directory = 'parabola_slope_social_test_random_NOINSTRUCTOR_2'
    os.mkdir(directory)

    # random_seeds = [8975, 91324,752324,1264183, 82376, 92835, 823975,147831, 234096, 2453, 2340554, 1234, 1321, 1457, 283, 2469,  12455,  2376324,
    #                 879363, 248979,43087926,564642,256874,344134,434634,34564,534645,344655,36455,31256]
    random_seeds = range(20)
    mode_ops = ['social']
    social_slopes = [1., 0.999999]

    groups1 = itertools.product(random_seeds, mode_ops, social_slopes)
    groups2 = itertools.product(random_seeds, ['autonomous'], [1.])
    for group in [groups1, groups2]:
        for idx,ops in enumerate(group):
            # Creating Agent ##
            system = System()
            if ops[1] is 'social':
                instructor = Instructor(thresh_slope=ops[2])
            else:
                instructor = None

            #interest model selection
            f_im_key = 'explauto_im'

            random_seed = ops[0]
            mode_ = ops[1]

            # Creating Models ##
            models = OBJECT()

            models.f_sm = model_(f_sm_key, system)
            models.f_ss = model_(f_ss_key, system)
            models.f_im = model_(f_im_key, system)
            models.f_cons = model_(f_cons_key, system)

            now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")
            file_prefix = directory + '/Parabola_Pool_' + str(idx) + '_' + now

            #  Creating Simulation object, running simulation and plotting experiments##
            evaluation_sim = SM_ModelEvaluation(system,
                                                models.f_sm, comp_func=comp_func)

            evaluation_sim.load_eval_dataset('../../Systems/datasets/instructor_parabola_1.h5')#('../../Systems/datasets/instructor_parabola_1.h5')'../../Systems/datasets/parabola_v2_dataset.h5'

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
            simulation.f_cons_key = f_cons_key
            simulation.f_im_key = f_im_key

            simulation.run(proprio=True)

            #evaluate sensorimotor
            evaluation_sim = SM_ModelEvaluation(system,
                                                simulation.models.f_sm,
                                                comp_func=comp_func,
                                                file_prefix=file_prefix + 'eval_whole_')

            # evaluate
            evaluation_sim.model.set_sigma_explo_ratio(0.)
            evaluation_sim.model.mode = 'exploit'

            evaluation_sim.load_eval_dataset('../../Systems/datasets/parabola_v2_dataset.h5')
            evaluation_sim.model.set_sigma_explo_ratio(0.)
            val_data = evaluation_sim.evaluate(space='sensor', saveData=True)

            evaluation_sim = SM_ModelEvaluation(system,
                                                simulation.models.f_sm,
                                                comp_func=comp_func,
                                                file_prefix=file_prefix + 'eval_social_')

            # evaluate
            evaluation_sim.model.set_sigma_explo_ratio(0.)
            evaluation_sim.model.mode = 'exploit'

            evaluation_sim.load_eval_dataset('../../Systems/datasets/instructor_parabola_1.h5')
            evaluation_sim.model.set_sigma_explo_ratio(0.)
            val_data = evaluation_sim.evaluate(space='sensor', saveData=True)

            if mode_ is 'social':
                import numpy as np
                np.savetxt(file_prefix + '_instructor_thresh.txt', instructor.unit_threshold)
            del simulation
            del models
            del evaluation_sim

    print('FINITO')