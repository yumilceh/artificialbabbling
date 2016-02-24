'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''

from DataTemplates.TabularData import TabularData
class SM_SS_Data(object):
    '''
    classdocs
    '''
    def __init__(self, Agent):
        self.motor_data=TabularData(self.motor_names)
        self.sensor_data=TabularData(self.sensor_data)
        self.somato_data=TabularData(self.somato_data)
    
    def appendData(self,Agent):
        self.motor_data.appendData(Agent.motorCommand)
        self.sensor_data.appendData(Agent.sensorOutput)
        self.somato_data.appendData(Agent.somatoOutput)


        