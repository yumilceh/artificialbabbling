"""
Created on Feb 21, 2017

@author: Juan Manuel Acevedo Valle
"""

from SensorimotorExploration.Models.Interest.GMM_IM import GMM_IM
from SensorimotorExploration.Models.Somatomotor.GMM_SS import GMM_SS
from SensorimotorExploration.Models.Sensorimotor.ILGMM_SM import GMM_SM as IGMM_SM
from SensorimotorExploration.Models.Somatomotor.ILGMM_SS import GMM_SS as IGMM_SS

from SensorimotorExploration.Models.Sensorimotor.ILGMM_SM_new import GMM_SM as IGMM_SM_new

from SensorimotorExploration.Algorithm.utils.competence_funcs import comp_Baraglia2015_expl as comp_func_expl
from SensorimotorExploration.Algorithm.utils.competence_funcs import comp_Baraglia2015 as comp_func



from SensorimotorExploration.Models.Interest.ExplautoIM import explauto_IM as ea_IM
from SensorimotorExploration.Models.Random import Random
from SensorimotorExploration.Models.Sensorimotor.ExplautoSM import ExplautoSM as ea_SM
from SensorimotorExploration.Models.Sensorimotor.GMM_SM import GMM_SM
from SensorimotorExploration.Models.Somatomotor.ExplautoSS import ExplautoSS as ea_SS
from SensorimotorExploration.Models.Somatomotor.ExplautoCons import ExplautoCons as ea_cons

model_class = {'gmm_sm': GMM_SM,
               'gmm_ss': GMM_SS,
               'igmm_sm': IGMM_SM,
               'igmm_sm_new': IGMM_SM_new,
               'igmm_ss': IGMM_SS,
               'gmm_im': GMM_IM,
               'explauto_im': ea_IM,
               'explauto_sm': ea_SM,
               'explauto_ss': ea_SS,
               'explauto_cons': ea_cons,
               'random': Random}

models_params_list = {'gmm_sm': [28],
                      'gmm_ss': [28],
                      'igmm_sm': [],
                      'igmm_sm_new': [],
                      'igmm_ss': [],
                      'gmm_im': [10],
                      'explauto_sm': [],  # 'LWLR-BFGS', 'nearest_neighbor', 'WNN', 'LWLR-CMAES'
                      'explauto_ss': [],  # 'LWLR-BFGS', 'nearest_neighbor', 'WNN', 'LWLR-CMAES'
                      'explauto_cons': [],  # 'LWLR-BFGS', 'nearest_neighbor', 'WNN', 'LWLR-CMAES'
                      'explauto_im': [],
                      'random': []
                      }

models_params_dict = {'gmm_sm': {'sm_step': 400,
                                 'alpha': 0.1,
                                  'sigma_explo_ratio': 0.},
                      'gmm_ss': {'ss_step': 400,
                                 'alpha': 0.1},
                      'igmm_sm': {'min_components': 3,
                                  'max_step_components': 10,
                                  'max_components': 30,
                                  'sm_step': 400,
                                  'forgetting_factor': 0.1,
                                  'sigma_explo_ratio': 0.},
                      'igmm_sm_new': {'min_components': 3,
                                  'max_step_components': 10,
                                  'max_components': 30,
                                  'sm_step': 400,
                                  'forgetting_factor': 0.1,
                                  'sigma_explo_ratio': 0.},
                      'igmm_ss': {'min_components': 3,
                                  'max_step_components': 10,
                                  'max_components': 30,
                                  'ss_step': 400,
                                  'forgetting_factor': 0.1},
                      'gmm_im': {'im_step': 30,
                                 'im_samples': 800},
                      'explauto_sm': {'model_type': 'non_parametric', 'model_conf': {'fwd': 'WNN', 'inv': 'WNN',
                                                                                     'k':3, 'sigma':.5,
                                                                                     'sigma_explo_ratio':0.}},
                      'explauto_ss': {'model_type': 'non_parametric', 'model_conf': {'fwd': 'WNN', 'inv': 'WNN',
                                                                                     'k':3, 'sigma':1.,
                                                                                     'sigma_explo_ratio':0.1}},
                      'explauto_cons': {'model_type': 'non_parametric', 'model_conf': {'fwd': 'WNN', 'inv': 'WNN',
                                                                                     'k':3, 'sigma':1.,
                                                                                     'sigma_explo_ratio':0.1}},
                      'explauto_im': {'competence_func': comp_func_expl, 'model_type': 'tree'},
                      'random': {'mode': 'sensor'}
                      }


def model_(model_key, system, competence_func=None):
    if competence_func is None:
        return model_class[model_key](system, *models_params_list[model_key], **models_params_dict[model_key])
    else:
        return model_class[model_key](system, competence_func, *models_params_list[model_key],
                                      **models_params_dict[model_key])
