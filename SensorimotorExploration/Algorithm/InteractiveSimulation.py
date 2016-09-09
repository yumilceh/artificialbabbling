'''
Created on Aug 31, 2016

@author: Juan Manuel Acevedo Valle
'''
#---------------------------------------------------------------- import sys, os

'''
Currently this class is only prepared to work with two dimensional sensori spaces
'''
import numpy as np
import random

from DataManager.SimulationData import SimulationData
from Algorithm.CompetenceFunctions import get_competence_Moulin2013
from Algorithm.StorageDataFunctions import saveSimulationData, loadSimulationData
from DataManager.PlotTools import *
from matplotlib.pyplot  import draw, show

class PARAMS(object):        
    def __init__(self):
        pass
class ManualSimulation(object):
    '''
        This class uses data in order to estimate a sensorimotor model and evaluate it.
    '''
    def __init__(self,  agent, 
                        file_prefix = '',
                        n_experiments = 10):
        '''
            Initialize
        '''
        self.agent = agent;
        self.data = PARAMS()
        self.n_experiments = n_experiments
        
        self.files=PARAMS()
        self.files.file_prefix = file_prefix
        self.fig, self.axes = initializeFigure();
        self.fig, self.axes = agent.drawSystem(self.fig, self.axes)
        self.fig.show()
        
    def executeManualMotorCommands(self, saveData = False):   
        fig=self.fig
        ax=self.axes
        plt.figure(fig.number)
        plt.sca(ax)   
        
        
        #Validation against Training set
        motor_command = self.agent.motor_command        
        for i in range(len(motor_command)):
            motor_command[i] = float(input("Insert the element " + str(i) + " of the motor command vector: "))
            
        self.agent.setMotorCommand(motor_command)
        self.agent.executeMotorCommand_unconstrained()
        plt.plot(self.agent.sensorOutput[0], self.agent.sensorOutput[1], "*k")
        self.agent.executeMotorCommand()
        plt.plot(self.agent.sensorOutput[0], self.agent.sensorOutput[1], "ob")
        
        self.fig = fig
        self.axes = ax
        draw()
        
        self.executeManualMotorCommands()
            
        