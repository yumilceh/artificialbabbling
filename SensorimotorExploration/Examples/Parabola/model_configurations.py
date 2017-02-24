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
from SensorimotorExploration.Models.explauto_SM import explauto_SM as ea_SM
from SensorimotorExploration.Models.explauto_SS import explauto_SS as ea_SS

from SensorimotorExploration.Models.random_model import RandomModel as RdnM


model_class = {'gmm_sm': GMM_SM,
               'gmm_ss': GMM_SS,
               'igmm_sm': IGMM_SM,
               'igmm_ss': IGMM_SS,
               'gmm_im': GMM_IM,
               'explauto_im': ea_IM,
               'explauto_sm': ea_SM,
               'explauto_ss': ea_SS,
               'random': RdnM}


models_params_list = {'gmm_sm': [10],
                     'gmm_ss': [10],
                 'igmm_sm': [],
                 'igmm_ss': [],
                 'gmm_im': [10],
                 'explauto_sm': [],
                 'explauto_ss': [],
                 'explauto_im': [],
                 'random': []
                 }

models_params_dict = {'gmm_sm': {'sm_step': 50,
                            'alpha': 0.5},
                 'gmm_ss': {'ss_step': 50,
                            'alpha': 0.5},
                 'igmm_sm': {'k_sm_min': 3,
                             'k_sm_step': 5,
                             'k_sm_max': 10,
                             'sm_step': 50,
                             'alpha': 0.05,
                             'sm_all_samples': False},
                 'igmm_ss': {'k_ss_min': 3,
                             'k_ss_step': 5,
                             'k_ss_max': 10,
                             'ss_step': 50,
                             'alpha': 0.05,
                             'sm_all_samples': False},
                 'gmm_im': {'im_step': 30,
                            'im_samples': 800},
                 'explauto_sm': {'model_type': 'nearest_neighbor'},
                 'explauto_ss': {'model_type': 'nearest_neighbor'},
                 'explauto_im': {'model_type': 'discretized_progress'},
                 'random': {'model_type': 'motor'}
                 }


def model_(model_key, system, competence_func = None):
    if type(competence_func) == type(None):
        return model_class[model_key](system, *models_params_list[model_key], **models_params_dict[model_key])
    else:
        return model_class[model_key](system, competence_func, *models_params_list[model_key], **models_params_dict[model_key])
