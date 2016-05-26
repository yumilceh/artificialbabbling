'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''

from DataManager.DataTemplates.TabularData import TabularData
import gzip
import shutil
import os
import matplotlib.pyplot as plt
#----------------------------------------------------------- import pandas as pd

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
        self.motor_data.data.to_hdf('motor_data.h5')
        self.sensor_data.data.to_hdf('sensor_data.h5')
        self.sensor_goal_data.data.to_hdf('sensor_goal_data.h5')
        self.somato_data.data.to_hdf('somato_data.h5')
        self.competence_data.data.to_hdf('competence_data.h5')
        for data_file in ['motor_data.h5', 
                     'sensor_data.h5',
                      'sensor_goal_data.h5',
                      'somato_data.h5',
                      'competence_data.h5']:
            with open(data_file,'rb') as f_in, gzip.open(file_name,'wb') as f_out:
                shutil.copyfileobj(f_in,f_out)
            os.remove(data_file)

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
        
        if src2=='motor':
            y_name=motor_names[column2]
            data2=self.motor_data.data[[y_name]]
        elif src2=='sensor':
            y_name=sensor_names[column2]
            data2=self.sensor_data.data[[y_name]]
        elif src2=='somato':
            y_name=somato_names[column2]
            data2=self.somato_data.data[[y_name]]
        
        plt.figure(fig.number)
        plt.sca(axes)    
        plt.plot(data1,data2,color)
        return fig,axes
    
    def plotTemporalSimulatedData(self,fig,axes,src,column,color):
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
        
        
        plt.figure(fig.number)
        plt.sca(axes)    
        plt.plot(data,color)
        return fig,axes
            
        