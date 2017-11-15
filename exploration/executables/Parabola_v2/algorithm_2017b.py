"""
Created on May 17, 2017

@author: Juan Manuel Acevedo Valle
"""
import datetime

class OBJECT(object):
    def __init__(self):
        pass

if __name__ == '__main__':
    #  Adding the projects folder to the path##

    # sys.path.append("../../")

    #  Adding libraries##
    from exploration.systems.parabola import ParabolicRegion as System
    from exploration.systems.parabola import Instructor
    from exploration.algorithm.trash.algorithm_2017b import Algorithm
    from exploration.algorithm.evaluation import Evaluation
    from exploration.data.PlotTools import *
    from exploration.algorithm.utils.functions import generate_motor_grid

    from model_configurations import model_, comp_func


    proprio = True
    mode = 'autonomous'
    expl_space = 'somato'  # 'sensor' for salient and 'somato' for haptic

    if expl_space is 'sensor':
        f_im_key = 'explauto_im'
    else:
        f_im_key = 'explauto_im_som'

    # models
    f_sm_key = 'igmm_sm'
    f_ss_key = 'igmm_ss'
    f_cons_key = 'explauto_cons'

    val_file =  '../../systems/datasets/parabola_v2_dataset.h5'


    """
       'igmm_sm': IGMM_SM,
       'igmm_ss': IGMM_SS,
       'explauto_im': ea_IM,
       'explauto_im_som': ea_IM,
       'explauto_cons': ea_cons,
       'random':  RdnM
    """

    # To guarantee reproducible experiments
    random_seed = 1234  # 12455   #1234

    n_initialization = 100
    n_experiments = 2000
    n_save_data = np.nan  # np.nan to not save, -1 to save 5 times during exploration

    eval_step = 200

    # Creating Agent ##
    system = System()
    instructor = Instructor()

    # Creating models ##
    models = OBJECT()

    models.f_sm = model_(f_sm_key, system)
    models.f_cons = model_(f_cons_key, system)
    models.f_im = model_(f_im_key, system)
    models.f_ss = model_(f_ss_key, system)

    evaluation_sim = Evaluation(system,
                                models.f_sm, comp_func=comp_func)

    evaluation_sim.load_eval_dataset(val_file)

    #  Creating Simulation object, running simulation and plotting experiments##
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_")
    file_prefix = 'Parabola_Sim_' + now
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
                           sm_all_samples=False)

    simulation.mode = mode  # social or autonomous
    simulation.set_expl_space(space = expl_space)


    # del(models)

    simulation.f_sm_key = f_sm_key
    simulation.f_cons_key = f_cons_key
    simulation.f_im_key = f_im_key

    simulation.run(proprio=proprio)

    sim_data = simulation.data

    evaluation_sim.model.set_sigma_explo_ratio(0.)
    evaluation_sim.model.mode = 'exploit'
    val_data = evaluation_sim.evaluate(save_data=False)
    error_ = np.linalg.norm(val_data.sensor_goal.data.as_matrix() -
                            val_data.sensor.data.as_matrix(), axis=1)
    print("Mean evaluation error is {} (max: {}, min: {})".format(np.mean(error_),
                                                                  np.max(error_),
                                                                  np.min(error_)))

    #  Looking at the proprioceptive model
    cons_th = system.cons_threshold

    n_motor_samples = 2000

    m1, m2 = generate_motor_grid(system, n_motor_samples)

    proprio_val = []
    for m in zip(m1.flatten(), m2.flatten()):
        system.cons_out = 0.
        system.set_action(np.array([m[0], m[1]]))
        cons_pred = simulation.models.f_cons.predict_cons(system)
        system.execute_action()
        cons_res = system.cons_out
        # print("We predicted {} but got {}.".format(cons_pred, cons_res))
        system.execute_action_unconstrained()

        if cons_pred >= cons_th and cons_res >= cons_th:
            proprio_val += [[system.sensor_out[0], system.sensor_out[1], '.k']]

        if cons_pred >= cons_th > cons_res:
            proprio_val += [[system.sensor_out[0], system.sensor_out[1], 'xr']]

        if cons_pred < cons_th and cons_res < cons_th:
            proprio_val += [[system.sensor_out[0], system.sensor_out[1], '.b']]

        if cons_pred < cons_th <= cons_res:
            proprio_val += [[system.sensor_out[0], system.sensor_out[1], 'xk']]

    simulation.params.expl_space='somato'
    model_to_eva = simulation.select_expl_model()
    evaluation_sim = Evaluation(system,
                                model_to_eva,
                                comp_func=comp_func,
                                file_prefix=file_prefix + '_')
    evaluation_sim.load_eval_dataset('../../systems/datasets/parabola_v2_dataset.h5')

    val_ssm_data = evaluation_sim.evaluate(save_data=False, space=simulation.params.expl_space)
    val_ssm_data.cut_final_data()

    simulation.params.expl_space='sensor'
    model_to_eva = simulation.select_expl_model()
    evaluation_sim = Evaluation(system,
                                model_to_eva,
                                comp_func=comp_func,
                                file_prefix=file_prefix + '_')
    evaluation_sim.load_eval_dataset('../../systems/datasets/parabola_v2_dataset.h5')

    val_sm_data = evaluation_sim.evaluate(save_data=False, space=simulation.params.expl_space)
    val_sm_data.cut_final_data()


    from plot_results import show_results
    show_results(system, simulation, val_sm_data, val_ssm_data, proprio_val)