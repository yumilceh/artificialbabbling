'''
Created on Jan 24, 2017

@author: Juan Manuel Acevedo Valle
'''
from importlib import import_module
from Models.explauto_SM import generateConfigurationExplauto
import numpy as np

model_class_name = {'discretized_progress': 'DiscretizedProgress',
                    'tree': 'InterestTree',
                    'gmm_progress_beta': 'GmmInterest'}

model_src_name = {'discretized_progress': 'discrete_progress',
                    'tree': 'tree',
                    'gmm_progress_beta': 'gmm_progress'}
                
model_conf = {'discretized_progress': {'x_card': 1000,
                                       'win_size': 10,
                                       'eps_random': 0.3},
              'tree':                 {'max_points_per_region': 100,
                                       'max_depth': 20,
                                       'split_mode': 'best_interest_diff',
                                       'progress_win_size': 50,
                                       'progress_measure': 'abs_deriv_smooth',                                                     
                                       'sampling_mode': {'mode':'softmax', 
                                                         'param':0.2,
                                                         'multiscale':False,
                                                         'volume':True}},
              'gmm_progress_beta':    {'n_samples': 40,
                                       'n_components': 6}             
             }

    
class PARAMS(object):
    def __init__(self):
        pass

class explauto_IM(object):
    '''
    Implemented for non-parametric models
    '''
    def __init__(self, agent, model_type, competence_func, model_conf = model_conf):
        model_competence_conf = {'discretized_progress': {'measure': competence_func},
                                 'tree':{'competence_measure': lambda target,reached : competence_func(target, reached)},
                                 'gmm_progress_beta':    {'measure': competence_func}             
                                      }
        model_conf[model_type].update(model_competence_conf[model_type])    
        
        conf = generateConfigurationExplauto(agent)
        self.conf = conf

                #-------------------------------------- ['discretized_progress', IMPLEMENTED
                #------------------------------------------------------- 'tree', IMPLEMENTED
                #----------------------------------------------------- 'random', 
                #------------------------------------------ 'miscRandom_global',
                #------------------------------------------ 'gmm_progress_beta', IMPLEMENTED
                #------------------------------------------- 'miscRandom_local']
            
        InterestModel =  getattr(import_module('explauto.interest_model.' + model_src_name[model_type]),  model_class_name[model_type])      
        
        self.model = InterestModel(conf, conf.s_dims, **model_conf[model_type])
        
        self.params = PARAMS()
        self.params.im_step = 1 #only ok with non-parametric
       
    def train(self,simulation_data):  
        m = simulation_data.motor_data.data.iloc[-1]
        s = simulation_data.sensor_data.data.iloc[-1]  
        s_g =  simulation_data.sensor_goal_data.data.iloc[-1]
        self.model.update(np.hstack((m, s_g)) , np.hstack((m, s)))  
            
    def get_interesting_goal(self,agent):
        return self.model.sample()
        
           
    def get_interesting_goal_proprioception(self,agent,sm_model,ss_model):
        ''' Not implemented yet'''
        return get_interesting_goal(agent)
        
            
    def get_interesting_goals(self,agent,n_goals=1):
        s_g = np.zeros((n_goals,agent.n_sensor))
        for i in range(n_goals):
            s_g[i,:] = self.model.sample()
        pass
           
    
    