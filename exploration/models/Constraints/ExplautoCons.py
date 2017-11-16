"""
Created on May 17, 2017

@author: Juan Manuel Acevedo Valle
"""

from explauto.sensorimotor_model.sensorimotor_model import SensorimotorModel
from explauto.sensorimotor_model.non_parametric import NonParametric
import numpy as np


class PARAMS(object):
    def __init__(self):
        pass


class ExplautoCons(object):
    """
    Implemented for non-parametric models
    """
    def __init__(self, system, model_type='nearest_neighbor', model_conf ="default"):
        conf = generateConfigurationExplauto(system)
        self.conf = conf

        if model_type == 'non_parametric':
            self.model = NonParametric(conf, **model_conf)
        else:
            self.model = SensorimotorModel.from_configuration(conf, model_type, model_conf)

        self.set_sigma_explo(0.)  # For conssensory data we are not interested on exploratory noise
        self.params = PARAMS()
        self.params.cons_step=1 #only ok with non-parametric
        self.params.model_type = model_type
        self.params.model_conf = model_conf
       
    def predict_cons(self, system, motor_command=None):
        if motor_command is None:
            motor_command = system.motor_command  #s_g
        
        system.cons_prediction = self.model.forward_prediction(motor_command)
        return  system.cons_prediction.copy()

    def train(self, simulation_data):
        m = simulation_data.motor.get_last(1).iloc[-1]
        s = simulation_data.cons.get_last(1).iloc[-1]
        # print('Trainign with m {} and som {}'.format(m,s))
        self.model.update(m,s)

    def train_old(self, simulation_data):
        m = simulation_data.motor.data.iloc[-1]
        s = simulation_data.cons.data.iloc[-1]
        # print('Trainign with m {} and som {}'.format(m,s))
        self.model.update(m,s)
        
    def train_incremental(self, simulation_data):
        self.train(simulation_data)
        
    def set_sigma_explo_ratio(self, new_value):
        conf = self.conf
        self.model.sigma_expl = (conf.m_maxs - conf.m_mins) * float(new_value)

    def set_sigma_explo(self, new_sigma):
        self.model.sigma_expl = new_sigma


    def get_sigma_explo(self):
        return self.model.sigma_expl

    def generate_log(self):
        params_to_logs = ['cons_step','model_type','model_conf']
        log = 'cons_model: EXPLAUTO_CONS\n'

        for attr_ in params_to_logs:
            if hasattr(self.params, attr_):
                try:
                    attr_log = getattr(self.params, attr_).generate_log()
                    log += attr_ + ': {\n'
                    log += attr_log
                    log += '}\n'
                except IndexError:
                    print("INDEX ERROR in EXPLAUTO_CONS log generation")
                except AttributeError:
                    attr_tmp = getattr(self.params, attr_)
                    if isinstance(attr_tmp, dict):
                        log += attr_ + ': {'
                        for key in attr_tmp.keys():
                            log += key + ': ' + str(attr_tmp[key]) + ','
                        log += ('}\n')
                    else:
                        log += attr_ + ': ' + str(attr_tmp) + '\n'
        return log

    def save_model(self, file_name, data_file = 'None'):
         with open(file_name, "w") as log_file:
            log_file.write('data_file: ' + data_file + '\n')
            log_file.write(self.generate_log())


def load_model(system, file_name, data=None):
    from exploration.data.data import load_sim_h5
    import string
    conf = {}
    with open(file_name) as f:
        for line in f:
            line = line.replace('\n', '')
            (key, val) = string.split(line,': ', maxsplit=1)
            conf.update({key: val})
    if ':' in conf['model_conf']:
        model_conf = {}
        line = conf['model_conf'].replace('\n', '')
        line = line.replace(',}', '')
        line = string.split(line, ',')
        for line_ in line:
            (key, val) = string.split(line_, ': ', maxsplit=1)
            key = key.replace('{','')
            model_conf.update({key: val})
            if model_conf[key]:
                try:
                    model_conf[key] = float(model_conf[key])
                    if key == 'k':
                        model_conf[key] = int(model_conf[key])
                except ValueError:
                    pass
    conf['model_conf'] = model_conf
    model = ExplautoCons(system, model_type=conf['model_type'], model_conf =conf['model_conf'])
    data, foo = load_sim_h5(conf['data_file'])
    motor = data.motor.data
    cons = data.cons.data
    for i in range(len(cons.index)):
        model.model.update(motor.iloc[i], cons.iloc[i])
    return model


def generateConfigurationExplauto(system):
    conf = PARAMS()
    conf.m_maxs = system.max_motor_values
    conf.m_mins = system.min_motor_values
    conf.s_maxs = system.max_cons_values
    conf.s_mins = system.min_cons_values
   
    n_motor = system.n_motor
    n_sensor = system.n_cons
    
    conf.m_ndims = n_motor
    conf.s_ndims = n_sensor

    conf.m_dims = np.arange(0, n_motor, 1).tolist()
    conf.s_dims = np.arange(n_motor, n_motor+n_sensor, 1).tolist()

    conf.bounds = np.zeros((2, n_motor + n_sensor))
    conf.bounds[0,:] = np.array(np.hstack((conf.m_mins, conf.s_mins))).flatten()
    conf.bounds[1,:] = np.array(np.hstack((conf.m_maxs, conf.s_maxs))).flatten()
    return conf
    
    

    
    
    