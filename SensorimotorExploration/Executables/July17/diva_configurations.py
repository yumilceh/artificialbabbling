"""
Created on Feb 21, 2017

@author: Juan Manuel Acevedo Valle
"""

from SensorimotorExploration.Models.Interest.GMM_IM import GMM_IM
from SensorimotorExploration.Models.Somatomotor.GMM_SS import GMM_SS
from SensorimotorExploration.Models.Sensorimotor.ILGMM_SM import GMM_SM as IGMM_SM
# from SensorimotorExploration.Models.Somatomotor.ILGMM_SM import GMM_SS as IGMM_SS

# from SensorimotorExploration.Algorithm.utils.competence_funcs import comp_Baraglia2015_expl as comp_func_expl
# from SensorimotorExploration.Algorithm.utils.competence_funcs import comp_Baraglia2015 as comp_func
from SensorimotorExploration.Algorithm.utils.competence_funcs import comp_Moulin2013_expl as comp_func_expl
from SensorimotorExploration.Algorithm.utils.competence_funcs import comp_Moulin2013 as comp_func

from SensorimotorExploration.Models.Interest.ExplautoIM import explauto_IM as ea_IM
from SensorimotorExploration.Models.Random import Random
from SensorimotorExploration.Models.Sensorimotor.ExplautoSM import ExplautoSM as ea_SM
from SensorimotorExploration.Models.Sensorimotor.GMM_SM import GMM_SM
from SensorimotorExploration.Models.Constraints.ExplautoCons import ExplautoCons as ea_cons

model_class = {'igmm_sm': IGMM_SM,
               'igmm_ss': IGMM_SM,
               'explauto_im': ea_IM,
               'explauto_im_som': ea_IM,
               'explauto_cons': ea_cons,
               'random': Random}

models_params_list = {'igmm_sm': [],
                      'igmm_ss': [],
                      'explauto_cons': [],
                      'explauto_im_som': [],
                      'explauto_im': [],
                      'random': []
                      }

models_params_dict = {'igmm_sm': {'min_components': 3,
                                  'max_step_components': 10,
                                  'max_components': 30,
                                  'sm_step': 400,
                                  'forgetting_factor': 0.1,
                                  'sigma_explo_ratio': 0.},
                      'igmm_ss': {'min_components': 3,
                                  'max_step_components': 10,
                                  'max_components': 30,
                                  'somato':True,
                                  'sm_step': 400,
                                  'forgetting_factor': 0.1},
                      'explauto_cons': {'model_type': 'non_parametric', 'model_conf': {'fwd': 'WNN', 'inv': 'WNN',
                                                                                       'k': 3, 'sigma': 1.,
                                                                                       'sigma_explo_ratio': 0.1}},
                      'explauto_im': {'competence_func': comp_func_expl, 'model_type': 'tree'},
                      'explauto_im_som': {'competence_func': comp_func_expl, 'model_type': 'tree','somato':True},
                      'random': {'mode': 'sensor'}
                      }


def model_(model_key, system, competence_func=None):
    if competence_func is None:
        return model_class[model_key](system, *models_params_list[model_key], **models_params_dict[model_key])
    else:
        return model_class[model_key](system, competence_func, *models_params_list[model_key],
                                      **models_params_dict[model_key])
