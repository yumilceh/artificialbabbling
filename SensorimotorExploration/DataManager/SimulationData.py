'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''

from DataTemplates.TabularData import TabularData
import matplotlib.pyplot as plt
#----------------------------------------------------------- import pandas as pd

class SimulationData(object):
    '''
    classdocs
    '''
    def __init__(self, Agent):
        self.motor_data=TabularData(Agent.motor_names)
        self.sensor_data=TabularData(Agent.sensor_names)
        self.somato_data=TabularData(Agent.somato_names)
    
    def appendData(self,Agent):
        self.motor_data.appendData(Agent.motorCommand)
        self.sensor_data.appendData(Agent.sensorOutput)
        self.somato_data.appendData(Agent.somatoOutput)

    def plotSimulatedData(self,src1,column1,src2,column2):
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
            
        plt.plot(data1,data2,"o")
        plt.show()
        
        
        