'''
Created on May 23, 2016

@author: Juan Manuel Acevedo Valle
'''
from ..DataManager.SimulationData import SimulationData

from ..Algorithm.utils.functions import get_random_motor_set, get_random_sensor_set
from ..Algorithm.utils.competence_funcs import comp_Moulin2013 as get_competence

import numpy as np
import numpy.linalg as linalg
from ..Algorithm.utils.data_storage_funcs import saveSimulationData

class PARAMS(object):
    def __init__(self):
        pass;

class MODELS(object):
    def __init__(self):
        pass;
        
class DATA(object):
    def __init__(self, alg):
        self.initialization_data_sm_ss = SimulationData(alg.agent)
        self.initialization_data_im = SimulationData(alg.agent)
        self.simulation_data = SimulationData(alg.agent)
   
class Algorithm_Random(object):

    '''
    classdocs
    '''

    def __init__(self,  agent,
                        models,
                        n_experiments = 1000,
                        random_seed = np.random.random((1,1)),
                        n_save_data = 100,
                        file_prefix='',
                        random_babbling='motor' #'sensor', 'motor'
                        ): 
        '''
        Constructor
        '''
        
        self.params = PARAMS();
    
        self.params.random_babbling=random_babbling
        self.params.n_experiments = n_experiments
        self.params.random_seed = random_seed
        
        self.params.n_save_data = n_save_data
        
        self.agent = agent
        self.models = models
    
        self.data = DATA(self)
        self.data.file_prefix=file_prefix
        
    def runNonProprioceptiveAlgorithm(self, n_motor_initialization=10):
        if self.params.random_babbling=='motor':
            self.runNonPA_motor()        
        elif self.params.random_babbling=='sensor' :
            self.runNonPA_sensor(n_motor_initialization)
        
    def runNonPA_motor(self):    
        n_save_data = self.params.n_save_data;
        n_experiments = self.params.n_experiments
        motor_commands =  get_random_motor_set(self.agent,
                                               n_experiments)    
        
        for i in range(n_experiments):
            self.agent.set_action(motor_commands[i, :])
            self.agent.executeMotorCommand()
            self.data.simulation_data.appendData(self.agent)
            if ((i+1)%self.models.f_sm.params.sm_step) == 0:
                self.models.f_sm.trainIncrementalLearning(self.data.simulation_data)
            if ((i+1)%self.models.f_ss.params.ss_step) == 0:
                self.models.f_ss.trainIncrementalLearning(self.data.simulation_data)
                print('Random motor babbling (Non-proprioceptive): Experiment: Training Models')
            print('Random motor babbling (Non-proprioceptive): Experiment: {} of {}'.format(i+1,n_experiments))
            if (np.mod(i,n_save_data) == 0):
                self.data.simulation_data.saveData(self.data.file_prefix +'simulation_data.h5')
        
        self.models.f_sm.trainIncrementalLearning(self.data.simulation_data)
        self.models.f_ss.trainIncrementalLearning(self.data.simulation_data)                        
        self.data.simulation_data.saveData(self.data.file_prefix + 'simulation_data.h5')
        saveSimulationData([self.data.file_prefix + 'simulation_data.h5'],'simulation_data.tar.gz')
        
    def runNonPA_sensor(self, n_motor_initialization):    
        n_save_data = self.params.n_save_data;
        n_experiments = self.params.n_experiments
        motor_commands =  get_random_motor_set(self.agent,
                                               n_motor_initialization)
        sensor_goals = get_random_sensor_set(self.agent,
                                             n_experiments)
        
        for i in range(n_motor_initialization):
            self.agent.set_action(motor_commands[i, :])
            self.agent.executeMotorCommand()
            self.data.initialization_data_sm_ss.appendData(self.agent)
            if ((i+1)%self.models.f_sm.params.sm_step) == 0:
                self.models.f_sm.trainIncrementalLearning(self.data.initialization_data_sm_ss)
            if ((i+1)%self.models.f_ss.params.ss_step) == 0:
                self.models.f_ss.trainIncrementalLearning(self.data.initialization_data_sm_ss)
                print('Random sensor babbling (Initialization-NP): Experiment: Training Models')
            print('Random sensor babbling (Initialization-NP): Experiment: {} of {}'.format(i+1,n_motor_initialization))
            if (np.mod(i,n_save_data) == 0):
                self.data.initialization_data_sm_ss.saveData(self.data.file_prefix +'initialization_data.h5')
                        
        self.data.initialization_data_sm_ss.saveData(self.data.file_prefix +'initialization_data.h5')
        self.models.f_sm.trainIncrementalLearning(self.data.initialization_data_sm_ss)
        self.models.f_ss.trainIncrementalLearning(self.data.initialization_data_sm_ss)
        
        for i in range(n_experiments):
            self.agent.sensor_goal = sensor_goals[i,:]
            self.models.f_sm.get_action(self.agent)
            self.agent.executeMotorCommand()
            get_competence(self.agent)
            self.data.simulation_data.appendData(self.agent)
            if ((i+1)%self.models.f_sm.params.sm_step) == 0:
                self.models.f_sm.trainIncrementalLearning(self.data.simulation_data)
            if ((i+1)%self.models.f_ss.params.ss_step) == 0:
                self.models.f_ss.trainIncrementalLearning(self.data.simulation_data)
                print('Random sensor babbling (Non-proprioceptive): Experiment: Training Models')
            print('Random sensor babbling (Non-proprioceptive): Experiment: {} of {}'.format(i+1,n_experiments))
            if (np.mod(i,n_save_data) == 0):
                self.data.simulation_data.saveData(self.data.file_prefix + 'simulation_data.h5')
                
        self.models.f_sm.trainIncrementalLearning(self.data.simulation_data)
        self.models.f_ss.trainIncrementalLearning(self.data.simulation_data)
        self.data.simulation_data.saveData(self.data.file_prefix +'simulation_data.h5')
        saveSimulationData([self.data.file_prefix + 'initialization_data.h5', self.data.file_prefix + 'simulation_data.h5'],'simulation_data.tar.gz')
        