"""
Created on June 24, 2017

@author: Juan Manuel Acevedo Valle
"""
import datetime

from diva_configurations import model_, comp_func
from exploration.algorithm.evaluation import Evaluation
from exploration.algorithm.trash.algorithm_2015 import OBJECT
from exploration.algorithm.algorithm2017 import Algorithm
#  Adding libraries##
from exploration.systems.Diva2017a import Diva2017a as System
from exploration.systems.Diva2017a import Instructor

# models
f_sm_key = 'igmm_sm'
f_ss_key = 'igmm_ss'
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
   'explauto_cons': ea_SS,
   'random':  RdnM
"""

if __name__ == '__main__':

    n_initialization = 100
    n_experiments = 20000
    n_save_data = 5000  # np.nan to not save, -1 to save 5 times during exploration

    eval_step = 2000 #np.nan to not evaluate

    random_seed =1234   # 1234 3487 9751
    proprio = False
    mode_ = 'autonomous'

    system = System()
    instructor = Instructor()#n_su=15)

    # Creating models ##
    models = OBJECT()
    models.f_sm = model_(f_sm_key, system)
    models.f_ss = model_(f_ss_key, system)
    models.f_im = model_(f_im_key, system)
    models.f_cons = model_(f_cons_key, system)

    # Creating Simulation object, running simulation and plotting experiments##
    # tree/DP Interest Model
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")

    file_prefix = 'test/motherese_test2_cmf_' + now

    evaluation_sim = Evaluation(system,
                                models.f_sm, comp_func=comp_func,
                                file_prefix=file_prefix)

    # evaluation_sim.set_eval_dataset('../../systems/datasets/german_dataset_somato.h5')
    evaluation_sim.set_eval_dataset(instructor.data)

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
    print('The work  is done!!')