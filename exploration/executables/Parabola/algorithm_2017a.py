"""
Created on Mar 8, 2017

@author: Juan Manuel Acevedo Valle
"""
import datetime

if __name__ == '__main__':
    #  Adding the projects folder to the path##

    # sys.path.append("../../")

    #  Adding libraries##
    from exploration.systems.trash.Parabola import ParabolicRegion as System
    from exploration.systems.trash.Parabola import Instructor
    from exploration.algorithm.algorithm_2017 import Algorithm as Algorithm
    from exploration.algorithm.trash.algorithm_2015 import OBJECT
    from exploration.algorithm.trash.ModelEvaluation import SM_ModelEvaluation
    from exploration.data.PlotTools import *
    from exploration.algorithm.utils.functions import generate_motor_grid

    from model_configurations import model_, comp_func

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

    # To guarantee reproducible experiments
    random_seed = 12455  # 12455   #1234

    n_initialization = 100
    n_experiments = 1000
    n_save_data = np.nan  # np.nan to not save, -1 to save 5 times during exploration

    eval_step = 200

    # random.seed(random_seed)
    # np_rnd.seed(random_seed)

    # Creating Agent ##
    system = System()
    instructor = Instructor()

    # Creating models ##
    models = OBJECT()

    models.f_sm = model_(f_sm_key, system)
    models.f_ss = model_(f_ss_key, system)
    models.f_im = model_(f_im_key, system)

    evaluation_sim = SM_ModelEvaluation(system,
                                        models.f_sm, comp_func=comp_func)

    evaluation_sim.loadEvaluationDataSet('../../systems/datasets/parabola_dataset_1.h5')

    proprio = True
    mode = 'autonomous'
    #  Creating Simulation object, running simulation and plotting experiments##
    now = datetime.datetime.now().strftime("Social_%Y_%m_%d_%H_%M_")
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

    # del(models)

    simulation.f_sm_key = f_sm_key
    simulation.f_ss_key = f_ss_key
    simulation.f_im_key = f_im_key

    simulation.run(proprio=proprio)

    sim_data = simulation.data

    evaluation_sim.model.set_sigma_explo_ratio(0.)
    evaluation_sim.model.mode = 'exploit'
    val_data = evaluation_sim.evaluate_model(saveData=False)
    error_ = np.linalg.norm(val_data.sensor_goal.data.as_matrix() -
                            val_data.sensor.data.as_matrix(), axis=1)
    print("Mean evaluation error is {} (max: {}, min: {})".format(np.mean(error_),
                                                                  np.max(error_),
                                                                  np.min(error_)))

    #  Looking at the proprioceptive model
    somato_th = system.somato_threshold

    n_motor_samples = 1000

    m1, m2 = generate_motor_grid(system, n_motor_samples)

    proprio_val = []
    for m in zip(m1.flatten(), m2.flatten()):
        system.somato_out = 0.
        system.set_action(np.array([m[0], m[1]]))
        somato_pred = simulation.models.f_ss.predict_somato(system)
        system.executeMotorCommand()
        somato_res = system.somato_out
        # print("We predicted {} but got {}.".format(somato_pred, somato_res))
        system.executeMotorCommand_unconstrained()

        if somato_pred >= somato_th and somato_res >= somato_th:
            proprio_val += [[system.sensor_out[0], system.sensor_out[1], '.k']]

        if somato_pred >= somato_th > somato_res:
            proprio_val += [[system.sensor_out[0], system.sensor_out[1], 'xr']]

        if somato_pred < somato_th and somato_res < somato_th:
            proprio_val += [[system.sensor_out[0], system.sensor_out[1], '.b']]

        if somato_pred < somato_th <= somato_res:
            proprio_val += [[system.sensor_out[0], system.sensor_out[1], 'xk']]

    evaluation_sim = SM_ModelEvaluation(system,
                                        simulation.models.f_sm,
                                        comp_func=comp_func,
                                        file_prefix=file_prefix + 'social_')
    evaluation_sim.loadEvaluationDataSet('../../systems/datasets/instructor_parabola_1.h5')
    val_data = evaluation_sim.evaluate_model(saveData=False)
    val_data.cut_final_data()


    from plot_results import show_results
    show_results(system, simulation, val_data, proprio_val)