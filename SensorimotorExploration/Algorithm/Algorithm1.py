'''
Created on May 23, 2016

@author: Juan Manuel Acevedo Valle
'''
from DataManager.SimulationData import SimulationData
from Models.GMM_SM import GMM_SM
from Models.GMM_SS import GMM_SS
from Models.GMM_IM import GMM_IM
from Algorithm.SupportFunctions import get_random_motor_set

import numpy as np
import numpy.linalg as linalg


class Algorithm1(object):


    '''
    classdocs
    '''

    def __init__(self,  agent, 
                        n_initialization_experiments=100,
                        n_experiments=100000,
                        random_seed=np.random.random((1,1)),
                        k_sm=28,
                        k_ss=28,
                        k_im=15,
                        g_im_initialization_method='non-zero'):
        '''
        Constructor
        '''
        self.agent=agent;
        self.n_initialization_experiments=n_initialization_experiments
        self.n_experiments=n_experiments
        self.random_seed=random_seed
        
        self.init_motor_commands=get_random_motor_set(agent,n_initialization_experiments)
        self.g_im_initialization_method=g_im_initialization_method
        self.gmm_sm=GMM_SM(agent,k_sm)
        self.gmm_ss=GMM_SS(agent,k_ss)
        self.gmm_im=GMM_IM(agent,k_im)        
            
        self.initialization_data_sm_ss=SimulationData(agent);
        self.initialization_data_im=SimulationData(agent);
        self.simulation_data=SimulationData(agent);
        
    def runNonProprioceptiveAlgorithm(self):
        n_init=self.n_initialization_experiments
        motor_commands=self.init_motor_commands
        agent=self.agent
        for i in range(n_init):
            agent.setMotorCommand(motor_commands[i,:])
            agent.getMotorDynamics()
            agent.executeMotorCommand()
            self.initialization_data_sm_ss.appendData(agent)
            print('Algorithm 1 (Non-proprioceptive), Line 1: Initialize G_SM and G_SS, experiment: {} of {}'.format(i,n_init))
        self.gmm_sm.train(self.initialization_data_sm_ss)
        self.gmm_ss.train(self.initialization_data_sm_ss)
        print('Algorithm 1 (Non-proprioceptive), Line 1: Initialize G_SM and G_SS, experiment {} of {}'.format(i,n_init))
        
        g_im_initialization_method=self.g_im_initialization_method
        if (g_im_initialization_method=='non-zero'):
            sensor_goals=self.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                print('Algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i,n_init))
                if(linalg.norm(sensor_goals[i])>0):
                    agent.sensor_goal=sensor_goals[i]
                    self.gmm_sm.getMotorCommand(agent)
                    agent.getMotorDynamics()
                    agent.executeMotorCommand()
                    get_competence_Moulin2013(agent)
                    self.initialization_data_im.appendData(agent)
                    print('Algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, Non-null sensory result considered ')
        
        self.gmm_im.train(self.initialization_data_im)
    
        n_experiments=self.n_experiments
        for i in range(n_experiments):
            agent.sensor_goal=self.gmm_im.get_interesting_goal()
            self.gmm_sm.getMotorCommand(agent)
            agent.getMotorDynamics()
            agent.executeMotorCommand()
            get_competence_Moulin2013(agent)
            self.simulation_data.appendData(agent)
            if ((i+1)%100)==0:
                self.gmm_im.train(self.simulation_data)
                self.gmm_sm.train(self.simulation_data)
                self.gmm_ss.train(self.simulation_data)
                print('Algorithm 1 (Non-proprioceptive), Line 4-22: Experiment: Training Models')
            print('Algorithm 1 (Non-proprioceptive), Line 4-22: Experiment: {} of {}'.format(i,n_experiments))


        
        
        
        

        
    
            
