'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''

from DataManager.DataTemplates.TabularData import TabularData
#===============================================================================
# import gzip
# import shutil
# import os
#===============================================================================
import matplotlib.pyplot as plt
#----------------------------------------------------------- import pandas as pd
from DataManager.PlotTools import movingAverage
import numpy as np


class SimulationData(object):
    '''
    classdocs
    '''
    def __init__(self, Agent):
        self.motor_data=TabularData(Agent.motor_names)
        self.sensor_data=TabularData(Agent.sensor_names)
        self.sensor_goal_data=TabularData(Agent.sensor_names)
        self.somato_data=TabularData(Agent.somato_names)
        self.competence_data=TabularData(['competence'])
    
    def appendData(self,Agent):
        self.motor_data.appendData(Agent.motor_command)
        self.sensor_data.appendData(Agent.sensorOutput)
        self.sensor_goal_data.appendData(Agent.sensor_goal)
        self.somato_data.appendData(Agent.somatoOutput)
        self.competence_data.appendData(Agent.competence_result)
        
    def saveData(self,file_name):
        self.motor_data.data.to_hdf(file_name,'motor_data')
        self.sensor_data.data.to_hdf(file_name,'sensor_data')
        self.sensor_goal_data.data.to_hdf(file_name,'sensor_goal_data')
        self.somato_data.data.to_hdf(file_name,'somato_data')
        self.competence_data.data.to_hdf(file_name,'competence_data')
       
    def cutData(self, agent, start, stop):
        simulationdata_tmp=SimulationData(agent)
        simulationdata_tmp.motor_data.data=self.motor_data.data.iloc[start:stop]
        simulationdata_tmp.sensor_data.data=self.sensor_data.data.iloc[start:stop]
        simulationdata_tmp.somato_data.data=self.somato_data.data.iloc[start:stop]
        simulationdata_tmp.competence_data.data=self.competence_data.data.iloc[start:stop]
        return simulationdata_tmp
    
    def mixDataSets(self, agent, sim_data_2):
        sim_data_1 = SimulationData(agent)
        sim_data_1.motor_data.data = self.motor_data.data.append(sim_data_2.motor_data.data)
        sim_data_1.sensor_data.data = self.sensor_data.data.append(sim_data_2.sensor_data.data)
        sim_data_1.sensor_goal_data.data = self.sensor_goal_data.data.append(sim_data_2.sensor_data.data)
        sim_data_1.somato_data.data = self.somato_data.data.append(sim_data_2.somato_data.data)
        sim_data_1.competence_data.data = self.competence_data.data.append(sim_data_2.competence_data.data)
        return sim_data_1
        
        
    def plotSimulatedData2D(self,fig,axes,src1,column1,src2,column2,color):
        motor_names=list(self.motor_data.data.columns.values)
        sensor_names=list(self.sensor_data.data.columns.values)
        somato_names=list(self.somato_data.data.columns.values)
        if src1=='motor':
            x_name=motor_names[column1]
            data1=self.motor_data.data[[x_name]]
        elif src1=='sensor':
            x_name=sensor_names[column1]
            data1=self.sensor_data.data[[x_name]]
        elif src1=='somato':
            x_name=somato_names[column1]
            data1=self.somato_data.data[[x_name]]
        elif src1=='sensor_goal':
            x_name=sensor_names[column1]
            data1=self.sensor_goal_data.data[[x_name]]
        
        if src2=='motor':
            y_name=motor_names[column2]
            data2=self.motor_data.data[[y_name]]
        elif src2=='sensor':
            y_name=sensor_names[column2]
            data2=self.sensor_data.data[[y_name]]
        elif src2=='somato':
            y_name=somato_names[column2]
            data2=self.somato_data.data[[y_name]]
        elif src2=='sensor_goal':
            y_name=sensor_names[column2]
            data2=self.sensor_goal_data.data[[y_name]]
        
        plt.figure(fig.number)
        plt.sca(axes)    
        plt.plot(data1,data2,color)
        return fig,axes
    
    def plotTemporalSimulatedData(self,fig,axes,src,column,color,moving_average=0):
        motor_names=list(self.motor_data.data.columns.values)
        sensor_names=list(self.sensor_data.data.columns.values)
        somato_names=list(self.somato_data.data.columns.values)
        if src=='motor':
            x_name=motor_names[column]
            data=self.motor_data.data[[x_name]]
        elif src=='sensor':
            x_name=sensor_names[column]
            data=self.sensor_data.data[[x_name]]
        elif src=='somato':
            x_name=somato_names[column]
            data=self.somato_data.data[[x_name]]
        elif src=='competence':
            data=self.competence_data.data[['competence']]
        elif src=='sensor_goal':
            x_name=sensor_names[column]
            data=self.sensor_goal_data.data[[x_name]]
        elif src=='error':    
            data = np.linalg.norm(self.sensor_goal_data.data - self.sensor_data.data, axis = 1)
            
        elif src=='error_log':
            data=np.log(self.competence_data.data[['competence']])
        
        if moving_average>0:
            try:
                data=movingAverage(data.as_matrix(),moving_average)
            except AttributeError:
                data=movingAverage(data, moving_average)

        plt.figure(fig.number)
        plt.sca(axes)    
        plt.plot(data,color)
        return fig,axes
            
        