'''
Created on Feb 5, 2016
This sensorimor system defines the DIVA agent used for the CCIA 2015's paper
@author: Juan Manuel Acevedo Valle
'''

#import sys
#import wave
import math
import numpy as np

from explauto import Environment

class SimpleArm:
    
    def __init__(self):
        self.environment = Environment.from_configuration('simple_arm', 'high_dim_high_s_range') #'default'
        conf = self.environment.conf
        

        n_motor = conf.m_ndims
        n_sensor = conf.s_ndims
        
        motor_names = []   
        sensor_names = [] 
        for i in range(n_motor): motor_names.append('M' + str(i))
        for i in range(n_sensor): sensor_names.append('S'+ str(i))
        somato_names = ['P1']
    
        n_somato = 1
        min_motor_values = np.array(conf.m_mins)
        max_motor_values = np.array(conf.m_maxs)
        
        min_sensor_values = np.array(conf.s_mins)
        max_sensor_values = np.array(conf.s_maxs)
        
        min_somato_values = np.array([0])
        max_somato_values = np.array([1])
        somato_threshold = np.array([0.6])

        self.n_motor = n_motor
        self.n_sensor = n_sensor
        self.n_somato = n_somato
        self.motor_names = motor_names
        self.sensor_names = sensor_names
        self.somato_names = somato_names
        
        self.min_motor_values = min_motor_values
        self.max_motor_values = max_motor_values
        self.min_sensor_values = min_sensor_values
        self.max_sensor_values = max_sensor_values
        self.min_somato_values = min_somato_values
        self.max_somato_values = max_somato_values
        self.somato_threshold = somato_threshold 

        self.motor_command =np.array( [0.0] * n_motor )
        self.sensorOutput = np.array( [0.0] * n_sensor )
        self.sensor_goal = np.array([ 0.0] * n_sensor )
        self.somatoOutput = np.array( [0.0] * n_somato )
        self.competence_result = 0.0;
        
    def setMotorCommand(self,motor_command):
        self.motor_command = motor_command    
        
    def executeMotorCommand(self):
        self.somatoOutput = 0.0
        self.sensorOutput = self.environment.update(self.motor_command)
        