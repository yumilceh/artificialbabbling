"""
Created on May 17, 2017

@author: Juan Manuel Acevedo Valle
"""

from exploration.algorithm.utils.competence_funcs import comp_Moulin2013_expl as comp_func_expl
from exploration.algorithm.utils.competence_funcs import comp_Moulin2013 as comp_func

# from exploration.models.sensorimotor.ExplautoSM import ExplautoSM as ea_SM
# from exploration.models.sensorimotor.GMM_SM import GMM_SM
from exploration.models.Constraints.ExplautoCons import ExplautoCons as ea_cons
from exploration.models.Interest.ExplautoIM import explauto_IM as ea_IM
from exploration.models.Interest.Random import Random
# from exploration.models.Interest.GMM_IM import GMM_IM
# from exploration.models.Somatomotor.GMM_SS import GMM_SS
from exploration.models.sensorimotor.IGMM_SM import GMM_SM as IGMM_SM
# from exploration.models.Somatomotor.ILGMM_SS import GMM_SS as IGMM_SS
from exploration.models.sensorimotor.trash.ILGMM_SM2 import GMM_SM as ILGMM_old
from igmm import DynamicParameter
from copy import copy,deepcopy
# from exploration.algorithm.utils.competence_funcs import comp_Baraglia2015_expl as comp_func_expl
# from exploration.algorithm.utils.competence_funcs import comp_Baraglia2015 as comp_func

model_class = {'igmm_sm': IGMM_SM,
               'igmm_old': ILGMM_old,
               'igmm_ss': IGMM_SM,
               'explauto_im': ea_IM,
               'explauto_im_som': ea_IM,
               'explauto_cons': ea_cons,
               'random': Random}

models_params_list = {'igmm_sm': [],
                      'igmm_ss': [],
                      'igmm_old': [],
                      'explauto_cons': [],  # 'LWLR-BFGS', 'nearest_neighbor', 'WNN', 'LWLR-CMAES'
                      'explauto_im': [],
                      'explauto_im_som':[],
                      'random': []
                      }

models_params_dict = {'igmm_ss': {'min_components': 3,
                                  'max_step_components': 10, #5
                                  'max_components': 80, #10
                                  'sm_step': 120,
                                  'somato': True,
                                  'forgetting_factor': 0.2, #OK: 0.2, 0.05
                                  'sigma_explo_ratio': 0.},
                      'igmm_sm': {'min_components': 3,
                                  'somato': False,
                                  'max_step_components': 5,
                                  'max_components': 20,
                                  'sm_step': 100,
                                  'forgetting_factor': DynamicParameter(
                                                        **{'function':'log',
                                                           'init':0.2,
                                                           'end':0.05,
                                                           'steps':162}
                                                            )},
                      'igmm_old': {'min_components': 3,
                                  'max_step_components': 5,
                                  'max_components': 20,
                                  'sm_step': 100,
                                  'forgetting_factor': 0.2},
                      'explauto_cons': {'model_type': 'non_parametric', 'model_conf': {'fwd': 'WNN', 'inv': 'WNN',
                                                                                     'k':3, 'sigma':1.,
                                                                                     'sigma_explo_ratio':0.1}},
                      'explauto_im': {'competence_func': comp_func_expl, 'model_type': 'discretized_progress'},
                      'explauto_im_som': {'competence_func': comp_func_expl, 'model_type': 'discretized_progress','somato':True},
                      'random': {'mode': 'sensor'}
                      }

def model_(model_key, system, competence_func=None):
    if competence_func is None:
        return deepcopy(model_class[model_key](system, *copy(models_params_list[model_key]), **copy(models_params_dict[model_key])))
    else:
        return deepcopy(model_class[model_key](system, competence_func, *copy(models_params_list[model_key]),
                                      **copy(models_params_dict[model_key])))