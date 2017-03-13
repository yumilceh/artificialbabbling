"""
Created on Mar 13, 2017

@author: Juan Manuel Acevedo Valle
"""

def get_config_log(alg, file_name):

    attribute_to_save = ['learner', 'instructor', 'f_sm_key',
                         'f_ss_key', 'f_im_key', ]
    
    learner = alg.learner.name
