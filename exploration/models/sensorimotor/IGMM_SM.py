'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from igmm import IGMM as GMM
import numpy as np
import pandas as pd
import copy 
from exploration.models.Sensorimotor import Sensorimotor
class PARAMS(object):
    def __init__(self):
        pass;
    
    
class GMM_SM(Sensorimotor):
    def __init__(self, system,
                       sm_step = 100,
                       min_components = 3,
                       max_step_components = 30,
                       max_components = 60,
                       a_split = 0.8,
                       forgetting_factor = 0.05,
                       sigma_expl_ratio = 0.0,
                       somato=False,
                       **kargs):
        Sensorimotor.__init__(self,system,sm_step=sm_step,somato=somato,sigma_expl_ratio=sigma_expl_ratio)
        if somato:
            n_sensor = system.n_somato
            sensor_names = system.somato_names
            sensor_space = 'somato'
        else:
            n_sensor = system.n_sensor
            sensor_names = system.sensor_names
            sensor_space = 'sensor'

        # self.params = PARAMS()
        # self.params.sensor_space = sensor_space
        # self.params.size_data = system.n_motor+ n_sensor
        # self.params.motor_names = system.motor_names
        # self.params.sensor_names = sensor_names
        #
        # self.params.n_motor=system.n_motor
        # self.params.n_sensor = n_sensor
        #
        # self.params.min_components = min_components
        # self.params.max_step_components = max_step_components
        # self.params.forgetting_factor = forgetting_factor
        # self.params.sm_step = sm_step
        #
        # self.delta_motor_values = system.max_motor_values - system.min_motor_values
        # self.sigma_expl = self.delta_motor_values * float(sigma_expl_ratio)
        # self.params.mode = 'exploit'

        m_dims = np.arange(0,self.params. n_motor, 1)
        s_dims = np.arange(self.params.n_motor, self.params.n_motor + self.params.n_sensor, 1)

        self.params.model=GMM(min_components = min_components,
                       max_step_components = max_step_components,
                       max_components = max_components,
                       a_split = a_split,
                       forgetting_factor = forgetting_factor,
                       x_dims = m_dims,
                       y_dims = s_dims)

    def set_forgetting_factor(self, value):
        self.params.forgetting_factor = value
        self.params.model.params['forgetting_factor'] = value

    def train(self, simulation_data):
        sensor_data = getattr(simulation_data, self.params.sensor_space)
        train_data_tmp = pd.concat([simulation_data.action.get_all(),
                                    sensor_data.get_all()], axis=1)
        self.params.model.train(train_data_tmp.as_matrix(columns=None))

    def train_incremental(self, simulation_data, all=False):
        sensor_data = getattr(simulation_data, self.params.sensor_space)
        if all:
            data = np.zeros((simulation_data.action.current_idx,
                             self.params.n_motor+self.params.n_sensor))
            data_m = simulation_data.action.get_all().as_matrix()
            data_s = sensor_data.get_all().as_matrix()
            data[:,:self.params.n_motor] = data_m
            data[:, self.params.n_motor:] = data_s
        else:
            data = np.zeros((self.params.sm_step,
                             self.params.n_motor+self.params.n_sensor))
            data_m = simulation_data.action.get_last(self.params.sm_step).as_matrix()
            data_s = sensor_data.get_last(self.params.sm_step).as_matrix()
            data[:,:self.params.n_motor] = data_m
            data[:, self.params.n_motor:] = data_s
        self.params.model.train(data)
    
    def get_action(self, system, sensor_goal=None):
        n_motor=system.n_motor
        n_sensor=self.params.n_sensor
        
        if sensor_goal is None:
            sensor_goal = getattr(system, self.params.sensor_space+'_goal')  #s_g
        
        m_dims=np.arange(0, n_motor, 1)
        s_dims= np.arange(n_motor, n_motor+n_sensor, 1)

        motor_command = self.params.model.infer(m_dims, s_dims, sensor_goal)

        motor_command = self.apply_sigma_expl(motor_command)
        # action = bound_action(system, action)
        system.motor_command = motor_command
        
        # return bound_action(system,self.params.model.predict(m_dims, s_dims, sensor_goal))  #Maybe this is wrong
        return motor_command.copy()

    def set_sigma_expl_ratio(self, new_value):
        self.params.sigma_expl = self.delta_motor_values * float(new_value)

    def set_sigma_expl(self, new_sigma):
        self.params.sigma_expl = new_sigma

    def get_sigma_expl(self):
        return copy.copy(self.params.sigma_expl)

    def generate_log(self):
        params_to_logs = ['sm_step',
                          'min_components',
                          'max_step_components',
                          'max_components',
                          'a_split',
                          'forgetting_factor',
                          'sigma_expl_ratio',
                          'sensor_space',
                          'n_motor',
                          'n_sensor']
        log = 'sm_model: IGMM_SM\n'

        for attr_ in params_to_logs:
            if hasattr(self.params, attr_):
                try:
                    attr_log = getattr(self.params, attr_).generate_log()
                    log += attr_ + ': {'
                    log += attr_log
                    log += '}\n'
                    log = log.replace('\n}', '}')
                except IndexError:
                    print("INDEX ERROR in ILGMM_SM log generation")
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

    def save(self, file_name):
         with open(file_name, "w") as log_file:
            log_file.write(self.generate_log())
         file_prefix = file_name.replace('.txt', '')
         self.params.model.save(file_prefix)

def load_model(system, file_name):
    from igmm import DynamicParameter
    from igmm import load_gmm
    import string
    conf = {}
    with open(file_name) as f:
        for line in f:
            line = line.replace('\n', '')
            (key, val) = string.split(line,': ', maxsplit=1)
            conf.update({key: val})
            if ':' in conf[key]:
                dict_ = {}
                line_ = conf[key].replace('\n', '')
                line_ = string.split(line_, ',')
                for line__ in line_:
                    (key_, val) = string.split(line__, ': ', maxsplit=1)
                    key_ = key_.replace('{','')
                    val = val.replace('}', '')
                    dict_.update({key_: val})
                    try:
                        dict_[key_] = float(dict_[key_])
                        if key_ == 'k':
                            dict_[key_] = int(dict_[key_])
                    except ValueError:
                        pass
                conf[key] = dict_
    forgetting_factor = DynamicParameter(**conf['forgetting_factor'])
    conf['forgetting_factor'] = forgetting_factor
    model = GMM_SM(system, **conf)
    gmm_ = load_gmm(file_name.replace('.txt',''))

    x_dims = np.arange(0, int(conf['n_motor']), 1)
    y_dims = np.arange(int(conf['n_motor']), int(conf['n_motor']) + int(conf['n_sensor']), 1)
    # print (m_dims)
    # print(s_dims)
    # gmm_.params['x_dims'] = m_dims
    # gmm_.params['y_dims'] = s_dims

    # gmm_.i.params['infer_fixed'] = True
    model.model.n_components = gmm_.n_components
    model.model.means_ = gmm_.means_
    model.model.weights_ = gmm_.weights_
    model.model.covariances_ = gmm_.covariances_

    # y_dims = gmm_.params['y_dims']
    # x_dims = gmm_.params['x_dims']

    SIGMA_YY_inv = np.zeros((gmm_.n_components, len(y_dims), len(y_dims)))
    SIGMA_XY = np.zeros((gmm_.n_components, len(x_dims), len(y_dims)))
    for k, (Mu, Sigma) in enumerate(zip(gmm_.means_, gmm_.covariances_)):
        Sigma_yy = Sigma[:, y_dims]
        Sigma_yy = Sigma_yy[y_dims, :]

        Sigma_xy = Sigma[x_dims, :]
        Sigma_xy = Sigma_xy[:, y_dims]
        Sigma_yy_inv = np.linalg.inv(Sigma_yy)

        SIGMA_YY_inv[k, :, :] = Sigma_yy_inv
        SIGMA_XY[k, :, :] = Sigma_xy

    model.model.SIGMA_YY_inv = SIGMA_YY_inv
    model.model.SIGMA_XY = SIGMA_XY
    # model = ExplautoCons(system, model_type=conf['model_type'], model_conf =conf['model_conf'])
    # data, foo = load_sim_h5(conf['data_file'])
    # action = data.action.data
    # cons = data.cons.data
    # for i in range(len(cons.index)):
    #     model.model.update(action.iloc[i], cons.iloc[i])
    return model


def bound_action(system, motor_command):
    n_motor=system.n_motor
    min_motor_values = system.min_motor_values
    max_motor_values = system.max_motor_values
    for i in range(n_motor):
        if (motor_command[i] < min_motor_values[i]):
            motor_command[i] = min_motor_values[i]
        elif (motor_command[i] > max_motor_values[i]):
            motor_command[i] = max_motor_values[i]
    return motor_command


"""
        for attr_ in params_to_logs:
            if hasattr(self.params, attr_):
                try:
                    attr_log = getattr(self.params, attr_).generate_log()
                    log+=(attr_ + ': {\n')
                    log+=(attr_log)
                    log+=('}\n')
                except IndexError:
                    print("INDEX ERROR in ILGMM log generation")
                except AttributeError:
                    log+=(attr_ + ': ' + str(getattr(self.params, attr_)) + '\n')
        return log
"""

