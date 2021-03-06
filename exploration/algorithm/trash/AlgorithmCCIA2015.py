'''
Created on May 23, 2016

@author: Juan Manuel Acevedo Valle
'''

from DataManager.SimulationData import SimulationData

from Algorithm.utils.RndSensorimotorFunctions import get_random_motor_set
from Algorithm.utils.CompetenceFunctions import get_competence_Moulin2013 as get_competence

#===============================================================================
# from algorithm.utils.CompetenceFunctions import comp_Baraglia2015 as get_competence
#===============================================================================

from Algorithm.ModelEvaluation import SM_ModelEvaluation
import numpy as np
import numpy.linalg as linalg
from Algorithm.utils.StorageDataFunctions import saveSimulationData

import copy 
from copy import deepcopy

class PARAMS(object):
    def __init__(self):
        pass;

class MODELS(object):
    def __init__(self):
        pass;
        
class DATA(object):
    def __init__(self, alg):
        self.init_motor_commands = get_random_motor_set(alg.agent,
                                                      alg.params.n_initialization_experiments)    
        self.initialization_data_sm_ss = SimulationData(alg.agent)
        self.initialization_data_im = SimulationData(alg.agent)
        self.simulation_data = SimulationData(alg.agent)
   
class Algorithm_CCIA2015(object):

    '''
    classdocs
    '''

    def __init__(self,  agent,
                        models,
                        n_initialization_experiments = 100,
                        n_experiments = 500000,
                        random_seed = np.random.random((1,1)),
                        g_im_initialization_method = 'non-zero', #'non-zero' 'all' 'non-painful'
                        n_save_data = 50000,
                        sm_all_samples = False,
                        evaluation = None,
                        file_prefix=''):
        '''
        Constructor
        '''

        self.params = PARAMS();
        self.params.n_initialization_experiments = n_initialization_experiments
        self.params.n_experiments = n_experiments
        self.params.random_seed = random_seed
        self.params.g_im_initialization_method = g_im_initialization_method
        self.params.n_save_data = n_save_data
        self.params.sm_all_samples = sm_all_samples
                
        self.agent = agent
        self.initialization_models = MODELS()
         
        self.models = models
    
        self.data = DATA(self)
        self.data.file_prefix=file_prefix
        
        self.evaluation = evaluation
        if not evaluation == None:
            self.evaluation_error = [1.0]
       
    def runNonProprioceptiveAlgorithm(self):
        n_init = self.params.n_initialization_experiments
        motor_commands = self.data.init_motor_commands
        for i in range(n_init):
            self.agent.set_action(motor_commands[i, :])
            self.agent.execute_action()
            self.data.initialization_data_sm_ss.append_data(self.agent)
            print('algorithm 1 (Non-proprioceptive), Line 1: Initialize G_SM and G_SS, experiment: {} of {}'.format(i,n_init))
        self.models.f_sm.train(self.data.initialization_data_sm_ss)
        self.models.f_ss.train(self.data.initialization_data_sm_ss)
        self.initialization_models.f_sm = self.models.f_sm.model.return_copy()
        self.initialization_models.f_ss = self.models.f_ss.model.return_copy()
        
        if not self.evaluation == None:
            self.evaluation.model = self.models.f_sm
            eval_data = self.evaluation.evaluate()
            error_ = np.linalg.norm(eval_data.sensor_goal_data.data - eval_data.sensor_data.data,axis = 1)
            self.evaluation_error = np.append(self.evaluation_error, np.mean(error_))

        print('algorithm 1 (Non-proprioceptive), Line 1: Initialize G_SM and G_SS, experiment {} of {}'.format(i,n_init))
        
        self.data.initialization_data_sm_ss.save_data(self.data.file_prefix + 'initialization_data_sm_ss.h5')
        
        g_im_initialization_method = self.params.g_im_initialization_method
        if (g_im_initialization_method == 'non-zero'):
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                print('algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i,n_init))
                if(linalg.norm(sensor_goals[i])>0):
                    self.agent.sensor_goal = sensor_goals[i]
                    self.models.f_sm.get_action(self.agent)
                    self.agent.execute_action()
                    get_competence(self.agent)
                    self.data.initialization_data_im.append_data(self.agent)
                    print('algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, Non-null sensory result considered ')
        elif (g_im_initialization_method == 'non-painful'):
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            proprio_data = self.data.initialization_data_sm_ss.somato_data.data.as_matrix()
            for i in range(n_init):
                print('algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i,n_init))
                if((proprio_data[i])==0):
                    self.agent.sensor_goal = sensor_goals[i]
                    self.models.f_sm.get_action(self.agent)
                    self.agent.execute_action()
                    get_competence(self.agent)
                    self.data.initialization_data_im.append_data(self.agent)
                    print('algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, Non-painful sensory result considered ')
        elif (g_im_initialization_method == 'all'):
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                print('algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i,n_init))
                self.agent.sensor_goal = sensor_goals[i]
                self.models.f_sm.get_action(self.agent)
                self.agent.execute_action()
                get_competence(self.agent)
                self.data.initialization_data_im.append_data(self.agent)
                print('algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, All sensory result considered ')
        self.data.initialization_data_im.save_data(self.data.file_prefix + 'initialization_data_im.h5')
        self.models.f_im.train(self.data.initialization_data_im)
        
        #=======================================================================
        # self.models.f_im.model.interactiveModel(self.models.f_im.get_train_data(self.data.initialization_data_im))
        #=======================================================================
        
        self.initialization_models.f_im = self.models.f_im.model.return_copy()
        
        
        
        n_save_data = self.params.n_save_data;
        n_experiments = self.params.n_experiments
        for i in range(n_experiments):
            self.agent.sensor_goal = self.models.f_im.get_goal(self.agent)
            self.models.f_sm.get_action(self.agent)
            self.agent.execute_action()
            get_competence(self.agent)
            self.data.simulation_data.append_data(self.agent)
            
            ''' Train Interest Model'''
            if ((i+1)%self.models.f_im.params.im_step) == 0:
                print('algorithm 1 (Non-proprioceptive), Line 4-22: Experiment: Training Model IM')
                if i < self.models.f_im.params.n_training_samples:
                    self.models.f_im.train(self.data.initialization_data_im.mix_datasets(self.agent, self.data.simulation_data))
                else:
                    self.models.f_im.train(self.data.simulation_data)
                
                
            ''' Train sensorimotor Model'''
            if ((i+1)%self.models.f_sm.params.sm_step) == 0:
                print('algorithm 1 (Non-proprioceptive), Line 4-22: Experiment: Training Model SM')
                if (i < n_init or self.params.sm_all_samples): ###BE CAREFUL WITH MEMORY
                    self.models.f_sm.train_incremental(
                                        self.data.simulation_data.mix_datasets(self.agent,
                                                                               self.data.initialization_data_im.mix_datasets(self.agent,
                                                                                                                             self.data.initialization_data_sm_ss)))
                else:
                    self.models.f_sm.train_incremental(self.data.simulation_data)
                if not self.evaluation == None:
                    self.evaluation.model = self.models.f_sm
                    eval_data = self.evaluation.evaluate()
                    error_ = np.linalg.norm(eval_data.sensor_goal_data.data - eval_data.sensor_data.data,axis = 1)
                    self.evaluation_error = np.append(self.evaluation_error, np.mean(error_))
                  
            ''' Train Somatosensory model'''   
            if ((i+1)%self.models.f_ss.params.ss_step) == 0:   
                print('algorithm 1 (Non-proprioceptive), Line 4-22: Experiment: Training Model SS')
                if (i < n_init or self.params.sm_all_samples): ###BE CAREFUL WITH MEMORY
                    self.models.f_ss.train_incremental(
                                        self.data.simulation_data.mix_datasets(self.agent,
                                                                               self.data.initialization_data_im.mix_datasets(self.agent,
                                                                                                                             self.data.initialization_data_sm_ss)))
                else:
                    self.models.f_sm.train_incremental(self.data.simulation_data)
                
            print('algorithm 1 (Non-proprioceptive), Line 4-22: Experiment: {} of {}'.format(i+1,n_experiments))
            if (np.mod(i,n_save_data) == 0):
                self.data.simulation_data.save_data(self.data.file_prefix + 'simulation_data.h5')
                
        self.data.simulation_data.save_data('simulation_data.h5')
        saveSimulationData([self.data.file_prefix + 'initialization_data_sm_ss.h5',
                           self.data.file_prefix + 'initialization_data_im.h5',
                           self.data.file_prefix + 'simulation_data.h5'],'simulation_data.tar.gz')

        
        
    def runProprioceptiveAlgorithm(self):
        n_init = self.params.n_initialization_experiments
        motor_commands = self.data.init_motor_commands
        for i in range(n_init):
            self.agent.set_action(motor_commands[i, :])
            self.agent.execute_action()
            self.data.initialization_data_sm_ss.append_data(self.agent)
            print('algorithm 1 (Proprioceptive), Line 1: Initialize G_SM and G_SS, experiment: {} of {}'.format(i+1,n_init))
        self.models.f_sm.train(self.data.initialization_data_sm_ss)
        self.models.f_ss.train(self.data.initialization_data_sm_ss)
        self.initialization_models.f_sm = self.models.f_sm.model.return_copy()
        self.initialization_models.f_ss = self.models.f_ss.model.return_copy()
        
        if not self.evaluation == None:
                    self.evaluation.model = self.models.f_sm
                    eval_data = self.evaluation.evaluate()
                    error_ = np.linalg.norm(eval_data.sensor_goal_data.data - eval_data.sensor_data.data,axis = 1)
                    self.evaluation_error = np.append(self.evaluation_error, np.mean(error_))
                    
        print('algorithm 1 (Proprioceptive), Line 1: Initialize G_SM and G_SS, experiment {} of {}'.format(i+1,n_init))
        
        self.data.initialization_data_sm_ss.save_data(self.data.file_prefix + 'initialization_data_sm_ss.h5')
        
        g_im_initialization_method = self.params.g_im_initialization_method
        if (g_im_initialization_method == 'non-zero'):
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                print('algorithm 1 (Proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i+1,n_init))
                if(linalg.norm(sensor_goals[i])>0):
                    self.agent.sensor_goal = sensor_goals[i]
                    self.models.f_sm.get_action(self.agent)
                    self.agent.execute_action()
                    get_competence(self.agent)
                    self.data.initialization_data_im.append_data(self.agent)
                    print('algorithm 1 (Proprioceptive), Line 2: Initialize G_IM, Non-null sensory result considered ')
        elif (g_im_initialization_method == 'non-painful'):
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            proprio_data = self.data.initialization_data_sm_ss.somato_data.data.as_matrix()
            for i in range(n_init):
                print('algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i,n_init))
                if((proprio_data[i])==0):
                    self.agent.sensor_goal = sensor_goals[i]
                    self.models.f_sm.get_action(self.agent)
                    self.agent.execute_action()
                    get_competence(self.agent)
                    self.data.initialization_data_im.append_data(self.agent)
                    print('algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, Non-painful sensory result considered ')
        
        elif (g_im_initialization_method == 'all'):
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                print('algorithm 1 (Proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i+1,n_init))
                self.agent.sensor_goal = sensor_goals[i]
                self.models.f_sm.get_action(self.agent)
                self.agent.execute_action()
                get_competence(self.agent)
                self.data.initialization_data_im.append_data(self.agent)
                print('algorithm 1 (Proprioceptive), Line 2: Initialize G_IM, All sensory result considered ')
        self.data.initialization_data_im.save_data(self.data.file_prefix + 'initialization_data_im.h5')
        self.models.f_im.train(self.data.initialization_data_im)
        self.initialization_models.f_im = self.models.f_im.model.return_copy()
        
        n_save_data = self.params.n_save_data;
        n_experiments = self.params.n_experiments
        for i in range(n_experiments):
            self.agent.sensor_goal = self.models.f_im.get_interesting_goal_proprioception(self.agent,self.models.f_sm,self.models.f_ss)
            self.models.f_im.activateAllComponents()
            self.models.f_sm.get_action(self.agent)
            self.agent.execute_action()
            get_competence(self.agent)
            self.data.simulation_data.append_data(self.agent)
            
            ''' Train Interest Model'''
            if ((i+1)%self.models.f_im.params.im_step) == 0:
                print('algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model IM')
                if i < self.models.f_im.params.n_training_samples:
                    self.models.f_im.train(self.data.initialization_data_im.mix_datasets(self.agent, self.data.simulation_data))
                else:
                    self.models.f_im.train(self.data.simulation_data)
                
                
            ''' Train sensorimotor Model'''
            if ((i+1)%self.models.f_sm.params.sm_step) == 0:
                print('algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model SM')
                if (i < n_init or self.params.sm_all_samples): ###BE CAREFUL WITH MEMORY
                    self.models.f_sm.train_incremental(
                                        self.data.simulation_data.mix_datasets(self.agent,
                                                                               self.data.initialization_data_im.mix_datasets(self.agent,
                                                                                                                             self.data.initialization_data_sm_ss)))
                else:
                    self.models.f_sm.train_incremental(self.data.simulation_data)
                if not self.evaluation == None:
                    self.evaluation.model = self.models.f_sm
                    eval_data = self.evaluation.evaluate()
                    error_ = np.linalg.norm(eval_data.sensor_goal_data.data - eval_data.sensor_data.data,axis = 1)
                    self.evaluation_error = np.append(self.evaluation_error, np.mean(error_))
                    
            ''' Train Somatosensory model'''   
            if ((i+1)%self.models.f_ss.params.ss_step) == 0:
                print('algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model SS')
                if (i < n_init or self.params.sm_all_samples): ###BE CAREFUL WITH MEMORY
                    self.models.f_ss.train_incremental(
                                        self.data.simulation_data.mix_datasets(self.agent,
                                                                               self.data.initialization_data_im.mix_datasets(self.agent,
                                                                                                                             self.data.initialization_data_sm_ss)))
                else:
                    self.models.f_sm.train_incremental(self.data.simulation_data)
                
                
            print('algorithm 1 (Proprioceptive), Line 4-22: Experiment: {} of {}'.format(i+1,n_experiments))
            if (np.mod(i,n_save_data) == 0):
                self.data.simulation_data.save_data(self.data.file_prefix + 'simulation_data.h5')
                
        self.data.simulation_data.save_data('simulation_data.h5')
        saveSimulationData([self.data.file_prefix + 'initialization_data_sm_ss.h5',
                           self.data.file_prefix + 'initialization_data_im.h5',
                           self.data.file_prefix + 'simulation_data.h5'],'simulation_data.tar.gz')

        
        
def returnEvaluationError(simulation):
    evaluation=SM_ModelEvaluation(simulation.system,
                                  10,
                                  simulation.models.f_sm)
        

        
    
            
