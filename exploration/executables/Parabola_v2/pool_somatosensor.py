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
    from exploration.systems.parabola import ParabolicRegion as System
    from exploration.systems.parabola import Instructor
    from exploration.algorithm.trash.algorithm_2017b import Algorithm
    from exploration.algorithm.trash.algorithm_2015 import OBJECT
    from exploration.algorithm.evaluation import Evaluation


    from model_configurations import model_, comp_func


    # models
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
    n_experiments = 2000
    n_save_data = 2000   # np.nan to not save, -1 to save 5 times during exploration

    eval_step = 200

    # random.seed(random_seed)
    # np_rnd.seed(random_seed)
    directory = 'experiment_2_ssm'
    os.mkdir(directory)


    random_seeds = [1234, 1321, 1457, 283, 2469, 147831, 234096, 2453, 2340554, 12455, 8975, 91324,752324,1264183, 82376, 92835, 823975, 2376324]
    proprio_ops = [True, False]
    mode_ops = ['autonomous']
    expl_space_ops = ['somato','sensor']

    for idx,ops in enumerate(itertools.product(random_seeds, proprio_ops, mode_ops, expl_space_ops)):
        # Creating Agent ##
        system = System()
        instructor = Instructor()

        #interest model selection
        expl_space = ops[3]
        if expl_space is 'somato':
            f_im_key = 'explauto_im_som'
        else:
            f_im_key = 'explauto_im'

        random_seed = ops[0]
        proprio = ops[1]
        mode_ = ops[2]

        # Creating models ##
        models = OBJECT()

        models.f_sm = model_(f_sm_key, system)
        models.f_ss = model_(f_ss_key, system)
        models.f_im = model_(f_im_key, system)
        models.f_cons = model_(f_cons_key, system)

        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_")
        file_prefix = directory + '/Parabola_Pool_' + str(idx) + '_' + now

        #  Creating Simulation object, running simulation and plotting experiments##
        evaluation_sim = Evaluation(system,
                                    models.f_sm, comp_func=comp_func)

        evaluation_sim.load_eval_dataset('../../systems/datasets/parabola_v2_dataset.h5')

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

        simulation.set_expl_space(space=expl_space)

        simulation.mode = mode_ # social or autonomous

        simulation.f_sm_key = f_sm_key
        simulation.f_ss_key = f_ss_key
        simulation.f_cons_key = f_cons_key
        simulation.f_im_key = f_im_key

        simulation.run(proprio=proprio)

        #evaluate sensorimotor
        evaluation_sim = Evaluation(system,
                                    simulation.models.f_sm,
                                    comp_func=comp_func,
                                    file_prefix=file_prefix + 'sensori_')

        # evaluate
        evaluation_sim.model.set_sigma_explo_ratio(0.)
        evaluation_sim.model.mode = 'exploit'

        evaluation_sim.load_eval_dataset('../../systems/datasets/parabola_v2_dataset.h5')
        evaluation_sim.model.set_sigma_explo_ratio(0.)
        val_data = evaluation_sim.evaluate(space='sensor', save_data=True)

        # evaluate somatosensorimotor
        evaluation_sim = Evaluation(system,
                                    simulation.models.f_ss,
                                    comp_func=comp_func,
                                    file_prefix=file_prefix + 'somato_')


        #evaluate
        evaluation_sim.model.set_sigma_explo_ratio(0.)
        evaluation_sim.model.mode = 'exploit'

        evaluation_sim.load_eval_dataset('../../systems/datasets/parabola_v2_dataset.h5')
        evaluation_sim.model.set_sigma_explo_ratio(0.)
        val_data = evaluation_sim.evaluate(space='somato', save_data=True)

        del simulation
        del models
        del evaluation_sim

    print('FINITO')