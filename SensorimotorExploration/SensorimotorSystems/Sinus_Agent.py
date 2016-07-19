'''
Created on Feb 5, 2016
This sensorimor system defines the DIVA agent used for the CCIA 2015's paper
@author: Juan Manuel Acevedo Valle
'''

#import sys
#import wave
import math
import numpy as np

#from matplotlib.pyplot import autoscale
#from matplotlib.animation import Animation
#from scipy.interpolate.interpolate_wrapper import block

class Sinus_Agent:
    
    def __init__(self):
        motor_names = ['M1']
        sensor_names = ['S1']
        somato_names = ['P1']
        n_motor = 1
        n_sensor = 1
        n_somato = 1
        min_motor_values = np.array([ 0.0 ])
        max_motor_values = np.array([ 2.0 * math.pi ])
        
        min_sensor_values = np.array([-1.0])
        max_sensor_values = np.array([1.0])
        
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
        
        self.motor_command =np.array( [0.0] * n_motor )
        self.sensorOutput = np.array( [0.0] * n_sensor )
        self.sensor_goal = np.array([ 0.0] * n_sensor )
        self.somatoOutput = np.array( [0.0] * n_somato )
        self.competence_result = 0.0;
        
    def setMotorCommand(self,motor_command):
        self.motor_command = motor_command    
        
    def executeMotorCommand(self):
        self.somatoOutput = 0.0
        self.sensorOutput = math.sin(self.motor_command)     