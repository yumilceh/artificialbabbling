'''
Created on May 23, 2016

@author: Juan Manuel Acevedo Valle
'''
from DataManager.SimulationData import SimulationData
from Models.GMM_SM import GMM_SM
from Models.GMM_SS import GMM_SS
from Models.GMM_IM import GMM_IM
from Algorithm.RndSensorimotorFunctions import get_random_motor_set
from Algorithm.CompetenceFunctions import get_competence_Moulin2013

import numpy as np
import numpy.linalg as linalg
from Algorithm.StorageDataFunctions import saveSimulationData

class PARAMS(object):
    def __init__(self):
        pass;

class MODELS(object):
    def __init__(self, alg):
        self.f_sm = GMM_SM(alg.agent,alg.params.k_sm)
        self.f_ss = GMM_SS(alg.agent,alg.params.k_ss)
        self.f_im = GMM_IM(alg.agent,alg.params.k_im)        
        
class DATA(object):
    def __init__(self, alg):
        self.init_motor_commands = get_random_motor_set(alg.agent,
                                                      alg.params.n_initialization_experiments)    
        self.initialization_data_sm_ss = SimulationData(alg.agent)
        self.initialization_data_im = SimulationData(alg.agent)
        self.simulation_data = SimulationData(alg.agent)
   
class Algorithm1(object):

    '''
    classdocs
    '''

    def __init__(self,  agent, 
                        n_initialization_experiments = 100,
                        n_experiments = 500000,
                        random_seed = np.random.random((1,1)),
                        k_sm = 28,
                        k_ss = 28,
                        k_im = 12,
                        g_im_initialization_method = 'non-zero',
                        n_save_data = 50000):
        '''
        Constructor
        '''
        
        
        
        self.params = PARAMS();
        self.params.n_initialization_experiments = n_initialization_experiments
        self.params.n_experiments = n_experiments
        self.params.random_seed = random_seed
        self.params.g_im_initialization_method = g_im_initialization_method
        self.params.n_save_data = n_save_data
        
        self.agent = agent
        
        self.params.k_sm = k_sm
        self.params.k_ss = k_ss
        self.params.k_im = k_im
        
        self.models = MODELS(self)

        self.data = DATA(self)
        
    def runNonProprioceptiveAlgorithm(self):
        n_init = self.params.n_initialization_experiments
        motor_commands = self.data.init_motor_commands
        for i in range(n_init):
            self.agent.setMotorCommand(motor_commands[i,:])
            self.agent.getMotorDynamics()
            self.agent.executeMotorCommand()
            self.data.initialization_data_sm_ss.appendData(self.agent)
            print('Algorithm 1 (Non-proprioceptive), Line 1: Initialize G_SM and G_SS, experiment: {} of {}'.format(i,n_init))
        self.models.f_sm.train(self.data.initialization_data_sm_ss)
        self.models.f_ss.train(self.data.initialization_data_sm_ss)
        print('Algorithm 1 (Non-proprioceptive), Line 1: Initialize G_SM and G_SS, experiment {} of {}'.format(i,n_init))
        
        self.data.initialization_data_sm_ss.saveData('initialization_data_sm_ss.h5')    
        
        g_im_initialization_method = self.params.g_im_initialization_method
        if (g_im_initialization_method == 'non-zero'):
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                print('Algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i,n_init))
                if(linalg.norm(sensor_goals[i])>0):
                    self.agent.sensor_goal = sensor_goals[i]
                    self.models.f_sm.getMotorCommand(self.agent)
                    self.agent.getMotorDynamics()
                    self.agent.executeMotorCommand()
                    get_competence_Moulin2013(self.agent)
                    self.data.initialization_data_im.appendData(self.agent)
                    print('Algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, Non-null sensory result considered ')
        self.data.initialization_data_im.saveData('initialization_data_im.h5')
        self.models.f_im.train(self.data.initialization_data_im)
        
        n_save_data = self.params.n_save_data;
        n_experiments = self.params.n_experiments
        for i in range(n_experiments):
            self.agent.sensor_goal = self.models.f_im.get_interesting_goal(self.agent)
            self.models.f_sm.getMotorCommand(self.agent)
            self.agent.getMotorDynamics()
            self.agent.executeMotorCommand()
            get_competence_Moulin2013(self.agent)
            self.data.simulation_data.appendData(self.agent)
            if ((i+1)%self.models.f_im.params.im_step) == 0:
                self.models.f_im.train(self.data.simulation_data)
            if ((i+1)%self.models.f_sm.params.sm_step) == 0:
                self.models.f_sm.trainIncrementalLearning(self.data.simulation_data)
            if ((i+1)%self.models.f_ss.params.ss_step) == 0:
                self.models.f_ss.trainIncrementalLearning(self.data.simulation_data)
                print('Algorithm 1 (Non-proprioceptive), Line 4-22: Experiment: Training Models')
            print('Algorithm 1 (Non-proprioceptive), Line 4-22: Experiment: {} of {}'.format(i,n_experiments))
            if (np.mod(i,n_save_data) == 0):
                self.data.simulation_data.saveData('simulation_data.h5')
                
        self.data.simulation_data.saveData('simulation_data.h5')
        saveSimulationData(['initialization_data_sm_ss.h5',
                           'initialization_data_im.h5',
                           'simulation_data.h5'],'simulation_data.tar.gz')

        
        
        
        

        
    
            
