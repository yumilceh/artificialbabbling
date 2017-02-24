"""
Created on Feb 21, 2017

@author: Juan Manuel Acevedo Valle
"""

from SensorimotorExploration.Models.GMM_IM import GMM_IM
from SensorimotorExploration.Models.GMM_SM import GMM_SM
from SensorimotorExploration.Models.GMM_SS import GMM_SS

from SensorimotorExploration.Models.ILGMM_SM import GMM_SM as IGMM_SM
from SensorimotorExploration.Models.ILGMM_SS import GMM_SS as IGMM_SS

from SensorimotorExploration.Models.explauto_IM import explauto_IM as ea_IM
from SensorimotorExploration.Models.ExplautoSM import ExplautoSM as ea_SM
from SensorimotorExploration.Models.ExplautoSS import ExplautoSS as ea_SS

from SensorimotorExploration.Models.Random import Random

from SensorimotorExploration.Algorithm.utils.CompetenceFunctions import comp_Moulin2013_expl as comp_func_expl

model_class = {'gmm_sm': GMM_SM,
               'gmm_ss': GMM_SS,
               'igmm_sm': IGMM_SM,
               'igmm_ss': IGMM_SS,
               'gmm_im': GMM_IM,
               'explauto_im': ea_IM,
               'explauto_sm': ea_SM,
               'explauto_ss': ea_SS,
               'random': Random}

models_params_list = {'gmm_sm': [15],
                      'gmm_ss': [15],
                      'igmm_sm': [],
                      'igmm_ss': [],
                      'gmm_im': [10],
                      'explauto_sm': [],  # 'LWLR-BFGS', 'nearest_neighbor', 'WNN', 'LWLR-CMAES'
                      'explauto_ss': ['nearest_neighbor'],  # 'LWLR-BFGS', 'nearest_neighbor', 'WNN', 'LWLR-CMAES'
                      'explauto_im': [],
                      'random': []
                      }

models_params_dict = {'gmm_sm': {'sm_step': 50,
                                 'alpha': 0.5},
                      'gmm_ss': {'ss_step': 50,
                                 'alpha': 0.5},
                      'igmm_sm': {'min_components': 3,
                                  'max_step_components': 5,
                                  'max_components': 10,
                                  'sm_step': 50,
                                  'forgetting_factor': 0.05},
                      'igmm_ss': {'min_components': 3,
                                  'max_step_components': 5,
                                  'max_components': 10,
                                  'ss_step': 50,
                                  'forgetting_factor': 0.05},
                      'gmm_im': {'im_step': 30,
                                 'im_samples': 800},
                      'explauto_sm': {'model_type': 'non_parametric', 'model_conf': {'fwd': 'WNN', 'inv': 'WNN',
                                                                                     'k':3, 'sigma':1.,
                                                                                     'sigma_explo_ratio':100}},
                      'explauto_ss': {'model_conf': 'default'},
                      'explauto_im': {'competence_func': comp_func_expl, 'model_type': 'discretized_progress'},
                      'random': {'mode': 'sensor'}
                      }


def model_(model_key, system, competence_func=None):
    if competence_func is None:
        return model_class[model_key](system, *models_params_list[model_key], **models_params_dict[model_key])
    else:
        return model_class[model_key](system, competence_func, *models_params_list[model_key],
                                      **models_params_dict[model_key])
