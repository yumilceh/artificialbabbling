'''
Created on Jun 3, 2016

@author: Juan Manuel Acevedo Valle
'''
from exploration.models.GeneralModels.IMLE import IMLE
import numpy as np

class PARAMS(object):
    def __init__(self):
        pass;

class IMLE_SM(): #Incremental Local Online 
    '''
    classdocs
    '''
    def __init__(self, agent, sm_step=400):
        self.params=PARAMS();
        self.params.size_data = agent.n_motor+agent.n_sensor
        self.params.sm_step = sm_step;
        self.params.motor_names = agent.motor_names;
        self.params.sensor_names = agent.sensor_names;
        self.params.n_motor = agent.n_motor;
        self.params.n_sensor = agent.n_sensor;
        
        self.params.in_dims = range(agent.n_motor)
        self.params.n_dims = agent.n_motor+agent.n_sensor
        self.params.out_dims = range(agent.n_motor,self.params.n_dims)
        self.params.min = (np.append(agent.min_motor_values, agent.min_sensor_values, axis = 0))
        self.params.max = (np.append(agent.max_motor_values, agent.max_sensor_values, axis = 0))
        
        #CONTROL OTHER PARAMETERS OF IMLE
        
        self.model=IMLE(self.params, mode='exploit')
        
        
        
    def train(self,simulation_data):
        self.trainIncrementalLearning(simulation_data)
        
    def trainIncrementalLearning(self,simulation_data):
        sm_step=self.params.sm_step
        print('Training IMLE_SM...')
        if (sm_step==1):
            x_ = simulation_data.motor.data.iloc[-1].as_matrix()
            y_ = simulation_data.sensor.data.iloc[-1].as_matrix()
            self.model.update(x_.astype(float),y_.astype(float))
            
        else:
            motor_data_size = len(simulation_data.motor.data.index)
            motor_data = simulation_data.motor.data[motor_data_size-sm_step:]
            sensor_data_size = len(simulation_data.sensor.data.index)
            sensor_data = simulation_data.sensor.data[sensor_data_size-sm_step:]
            for i in range(sm_step):
                x_ = motor_data.iloc[i].as_matrix()
                y_ = sensor_data.iloc[i].as_matrix()
                self.model.update(x_.astype(float),y_.astype(float))
        print('Training finished')
             
    
    def getMotorCommand(self,Agent):
        sensor_goal=Agent.sensor_goal  #s_g        
        Agent.motor_command=self.model.infer(self.params.out_dims,self.params.in_dims,sensor_goal.astype(float))
    