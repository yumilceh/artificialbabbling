'''
Created on Feb 5, 2016

@author: yumilceh
'''
from SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a
from DataManager.SimulationData import SimulationData
from Models.GMM_SM  import GMM_SM
from DataVisualization.DataVisualizer import DataVisualizer
import numpy as np


if __name__ == '__main__':
    
    diva_agent=Diva_Proprio2015a()
    
    n_vocalizations=1000;
    
    simulation_data=SimulationData(diva_agent);
    
    n_motor_commands=diva_agent.n_motor;
    
    motor_commands=np.random.random((n_vocalizations,n_motor_commands))
    
    gmm_sm=GMM_SM(diva_agent,28);
    
    for i in range(n_vocalizations):
        diva_agent.setMotorCommand(motor_commands[i,:])
        diva_agent.getMotorDynamics()
        diva_agent.vocalize()
        simulation_data.appendData(diva_agent)
        print(i)
     
    #===========================================================================
    # gmm_sm.train(simulation_data)
    #===========================================================================
    
    
    #===========================================================================
    # print(simulation_data.motor_data.data)
    # print(simulation_data.sensor_data.data)
    #===========================================================================
    gmm_sm.train(simulation_data)
    
    #------------------------------------------ print(gmm_sm.GMM.model.weights_)
    #-------------------------------------------- print(gmm_sm.GMM.model.means_)
    #------------------------------------------- print(gmm_sm.GMM.model.covars_)
    
    
    visualizer=DataVisualizer(diva_agent)
    simulation_data.plotSimulatedData('sensor', 0, 'sensor', 3)
    
    
          
    #===========================================================================
    # diva_agent.stop()
    # del(diva_agent)
    #===========================================================================
     
     
