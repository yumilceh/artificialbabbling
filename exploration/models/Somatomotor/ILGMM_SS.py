'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
import pandas as pd

import numpy as np

from exploration.models.GeneralModels.Trash.ILGMM_GREC import ILGMM as GMM


class PARAMS(object):
    def __init__(self):
        pass;


class GMM_SS(object):
    '''
    classdocs
    '''

    def __init__(self, agent,
                 ss_step=100,
                 min_components=3,
                 max_step_components=30,
                 max_components=60,
                 a_split=0.8,
                 forgetting_factor=0.05,
                 plot=False, plot_dims=[0, 1]):
        '''
        Constructor
        '''

        self.params = PARAMS()
        self.params.size_data = agent.n_motor + agent.n_somato
        self.params.motor_names = agent.motor_names
        self.params.somato_names = agent.sensor_names
        self.params.n_motor = agent.n_motor
        self.params.n_somato = agent.n_somato
        self.params.min_components = min_components
        self.params.max_step_components = max_step_components
        self.params.forgetting_factor = forgetting_factor
        self.params.ss_step = ss_step

        self.model = GMM(min_components=min_components,
                         max_step_components=max_step_components,
                         max_components=max_components,
                         a_split=a_split,
                         forgetting_factor=forgetting_factor,
                         plot=plot, plot_dims=plot_dims)

    def train(self, simulation_data):
        train_data_tmp = pd.concat([simulation_data.motor_data.data, simulation_data.somato_data.data], axis=1)
        self.model.train(train_data_tmp.as_matrix(columns=None))

    def trainIncrementalLearning(self, simulation_data):
        # ------------------------------------------- ss_step=self.params.ss_step
        # ----------------------------------------------- alpha=self.params.alpha
        # ------------ motor_data_size=len(simulation_data.action.data.index)
        # art=simulation_data.art.data[motor_data_size-ss_step:-1]
        # ---------- somato_data_size=len(simulation_data.somato.data.index)
        # somato=simulation_data.somato.data[somato_data_size-ss_step:-1]
        # ------------------- new_data=pd.concat([action,somato],axis=1)
        # ------------------ self.model.trainIncrementalLearning(new_data, alpha)
        train_data_tmp = pd.concat([simulation_data.motor_data.data, simulation_data.somato_data.data], axis=1)
        self.model.train(train_data_tmp.as_matrix(columns=None))

    def predictProprioceptiveEffect(self, Agent, motor_command=None):
        n_motor = Agent.n_motor;
        n_somato = Agent.n_somato;

        if motor_command == None:
            motor_command = Agent.motor_command  # s_g

        m_dims = np.arange(0, n_motor, 1)
        s_dims = np.arange(n_motor, n_motor + n_somato, 1)
        Agent.proprioceptive_prediction = self.model.predict(s_dims, m_dims, motor_command)
        return boundProprioceptivePrediction(Agent, self.model.predict(s_dims, m_dims, motor_command))


def boundProprioceptivePrediction(Agent, proprioceptive_prediction):
    n_somato = Agent.n_somato;
    min_somato_values = Agent.min_somato_values
    max_somato_values = Agent.max_somato_values
    somato_threshold = Agent.somato_threshold
    for i in range(n_somato):
        if ((proprioceptive_prediction[i] < min_somato_values[i]) or (
            proprioceptive_prediction[i] <= somato_threshold)):
            proprioceptive_decision = 0

        elif (
            (proprioceptive_prediction[i] > max_somato_values[i]) or (proprioceptive_prediction[i] > somato_threshold)):
            proprioceptive_decision = 1
    return proprioceptive_decision
