"""
Created on Feb 21, 2017

@author: Juan Manuel Acevedo Valle
"""

from SensorimotorExploration.Models.GMM_IM import GMM_IM
from SensorimotorExploration.Models.GMM_SS import GMM_SS
from SensorimotorExploration.Models.ILGMM_SM import GMM_SM as IGMM_SM
from SensorimotorExploration.Models.ILGMM_SS import GMM_SS as IGMM_SS

from SensorimotorExploration.Models.Interest.ExplautoIM import explauto_IM as ea_IM
from SensorimotorExploration.Models.Random import RandomModel as RdnM
from SensorimotorExploration.Models.Sensorimotor.ExplautoSM import ExplautoSM as ea_SM
from SensorimotorExploration.Models.Sensorimotor.GMM_SM import GMM_SM
from SensorimotorExploration.Models.Somatomotor.ExplautoSS import ExplautoSS as ea_SS

model_class = {'gmm_sm': GMM_SM,
               'gmm_ss': GMM_SS,
               'igmm_sm': IGMM_SM,
               'igmm_ss': IGMM_SS,
               'gmm_im': GMM_IM,
               'explauto_im': ea_IM,
               'explauto_sm': ea_SM,
               'explauto_ss': ea_SS,
               'random': RdnM}


models_params = {'gmm_sm': {'k_sm': 10,
                            'sm_step': 50,
                            'alpha_sm': 0.5,
                            'sm_all_samples': False},
                 'gmm_ss': {'k_ss': 10,
                            'ss_step': 50,
                            'alpha_ss': 0.5,
                            'ss_all_samples': False},
                 'igmm_sm': {'k_sm_min': 3,
                             'k_sm_step': 5,
                             'k_sm_max': 10,
                             'sm_step': 50,
                             'alpha_sm': 0.05,
                             'sm_all_samples': False},
                 'igmm_ss': {'k_ss_min': 3,
                             'k_ss_step': 5,
                             'k_ss_max': 10,
                             'ss_step': 50,
                             'alpha_ss': 0.05,
                             'sm_all_samples': False},
                 'gmm_im': {'k_im': 10,
                            'im_step': 30,
                            'im_samples': 800},
                 'explauto_sm': {'model_type': 'nearest_neighbor'},
                 'explauto_ss': {'model_type': 'nearest_neighbor'},
                 'explauto_im': {'model_type': 'discretized_progress'},
                 'random': {'model_type': 'art'}
                 }


def model_(model_key, system, competence_func = None):
    if type(competence_func) == type(None):
        return model_class[model_key](system, **models_params[model_key])
    else:
        return model_class[model_key](system, competence_func, **models_params[model_key])
