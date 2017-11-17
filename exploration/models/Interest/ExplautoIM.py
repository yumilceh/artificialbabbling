'''
Created on Jan 24, 2017

@author: Juan Manuel Acevedo Valle
'''
import numpy as np
from importlib import import_module

# from SensorimotorExploration.models.Sensorimotor.ExplautoSM import generateConfigurationExplauto

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

    
class OBJECT(object):
    def __init__(self):
        pass

class explauto_IM(object):
    '''
    Implemented for non-parametric models
    '''
    def __init__(self, system,competence_func,  model_type, model_conf = model_conf, somato=False):
        model_competence_conf = {'discretized_progress': {'measure': competence_func},
                                 'tree':{'competence_measure': lambda target,reached : competence_func(target, reached)},
                                 'gmm_progress_beta':    {'measure': competence_func}             
                                      }
        model_conf[model_type].update(model_competence_conf[model_type])    
        
        conf = generateConfigurationExplauto(system, somato=somato)
        self.conf = conf

                #-------------------------------------- ['discretized_progress', IMPLEMENTED
                #------------------------------------------------------- 'tree', IMPLEMENTED
                #----------------------------------------------------- 'random', 
                #------------------------------------------ 'miscRandom_global',
                #------------------------------------------ 'gmm_progress_beta', IMPLEMENTED
                #------------------------------------------- 'miscRandom_local']
            
        InterestModel =  getattr(import_module('explauto.interest_model.' + model_src_name[model_type]),  model_class_name[model_type])      
        
        self.model = InterestModel(conf, conf.s_dims, **model_conf[model_type])
        
        self.params = OBJECT()
        
        if somato:
            self.params.sensor_space = 'somato'
        else:
            self.params.sensor_space = 'sensor'

        self.params.model_type, self.params.model_conf = model_type, model_conf
        self.params.im_step = 1 #only ok with non-parametric
        self.params.n_training_samples = 1  # only ok with non-parametric

    def train(self,simulation_data): 
        sensor_data = getattr(simulation_data, self.params.sensor_space)
        m = simulation_data.motor.get_last(1).as_matrix()
        s = sensor_data.get_last(1).as_matrix()
        sensor_goal_data = getattr(simulation_data, self.params.sensor_space+'_goal')
        s_g =  sensor_goal_data.get_last(1).as_matrix()
        self.model.update(np.hstack((m, s_g))[0], np.hstack((m, s))[0])
            
    def get_goal(self):
        return self.model.sample()

    def get_goal_proprio(self, system, sm_model, cons_model, n_attempts = 10): #n_attempts = 30 worse behavior
        tmp_goal = self.get_goal()
        tmp_motor = sm_model.get_action(system, sensor_goal=tmp_goal)
        tmp_cons = cons_model.predict_cons(system, motor_command=tmp_motor)
        n_attempts -= 1
        if tmp_cons < system.cons_threshold or n_attempts == 0:
            return tmp_goal
        else:
            return self.get_goal_proprio(system, sm_model, cons_model, n_attempts=n_attempts)

    def get_goals(self, n_goals=1):
        s_g = np.zeros((n_goals,self.conf.n_sensor))
        for i in range(n_goals):
            s_g[i,:] = self.model.sample()
        pass

    def generate_log(self):
        params_to_logs = ['space', 'im_step', 'n_training_samples', 'model_type', 'model_conf']
        log = 'im_model: EXPLAUTO_IM\n'

        for attr_ in params_to_logs:
            if hasattr(self.params, attr_):
                try:
                    attr_log = getattr(self.params, attr_).generate_log()
                    log += attr_ + ': {'
                    log += attr_log
                    log += '}\n'
                except IndexError:
                    print("INDEX ERROR in ExplautoIM log generation")
                except AttributeError:
                    if isinstance(getattr(self.params, attr_), dict):
                        log += attr_ + ': {'
                        for key in getattr(self.params, attr_).keys():
                            log += key + ': ' + str(getattr(self.params, attr_)[key]) + ','
                        log += ('}\n')
                        log = log.replace(',}', '}')
                    else:
                        log += attr_ + ': ' + str(getattr(self.params, attr_)) + '\n'
        return log

def generateConfigurationExplauto(system, somato=False):
    conf = OBJECT()
    conf.m_maxs = system.max_motor_values
    conf.m_mins = system.min_motor_values
    if somato:
        conf.s_maxs = system.max_somato_values
        conf.s_mins = system.min_somato_values
        n_sensor = system.n_somato
    else:
        conf.s_maxs = system.max_sensor_values
        conf.s_mins = system.min_sensor_values
        n_sensor = system.n_sensor

    n_motor = system.n_motor

    conf.m_ndims = n_motor
    conf.s_ndims = n_sensor

    conf.m_dims = np.arange(0, n_motor, 1).tolist()
    conf.s_dims = np.arange(n_motor, n_motor + n_sensor, 1).tolist()

    conf.bounds = np.zeros((2, n_motor + n_sensor))
    conf.bounds[0, :] = np.array(np.hstack((conf.m_mins, conf.s_mins))).flatten()
    conf.bounds[1, :] = np.array(np.hstack((conf.m_maxs, conf.s_maxs))).flatten()
    return conf

