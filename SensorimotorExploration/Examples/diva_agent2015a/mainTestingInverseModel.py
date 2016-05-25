'''
Created on Feb 5, 2016

@author: yumilceh
'''
from SensorimotorExploration.SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a
from SensorimotorExploration.DataManager.SimulationData import SimulationData
from SensorimotorExploration.Models.GMM_SM  import GMM_SM
from SensorimotorExploration.DataVisualization.PlotTools import *
import numpy as np

if __name__ == '__main__':
    
    diva_agent=Diva_Proprio2015a()
    
    n_vocalizations=28;
    
    simulation_data=SimulationData(diva_agent);
    
    n_motor_commands=diva_agent.n_motor;
    
    motor_commands=np.random.random((n_vocalizations,n_motor_commands))
    
    gmm_sm=GMM_SM(diva_agent,28);
    
    for i in range(n_vocalizations):
        diva_agent.setMotorCommand(motor_commands[i,:])
        diva_agent.getMotorDynamics()
        diva_agent.vocalize()
        simulation_data.appendData(diva_agent)
        print('Random initialization, vocalization: {}'.format(i))

    gmm_sm.train(simulation_data)

    f,ax=initializeFigure();
    f,ax=simulation_data.plotSimulatedData(f,ax,'sensor', 0, 'sensor', 3,"ob")
    
    
    n_random_examples=5
    random_indexes=np.random.randint(n_vocalizations,size=n_random_examples)
    
    sensor_goals=simulation_data.sensor_data.data.as_matrix()
    sensor_goals=sensor_goals[random_indexes,:]
    
     # simulation_data.sensor_data.data.drop(simulation_data.sensor_data.data.index[:])
     # simulation_data.motor_data.data.drop(simulation_data.motor_data.data.index[:])
     # simulation_data.somato_data.data.drop(simulation_data.somato_data.data.index[:])
    simulation_data=SimulationData(diva_agent);

    #----------------------------------------------------- print(random_indexes)
#------------------------------------------------------------------------------ 
    #------------------------------------------------------- print(sensor_goals)
    print(diva_agent.motor_command)
    
    for i in range(n_random_examples):
        diva_agent.sensor_goal=sensor_goals[i]
        gmm_sm.getMotorCommand(diva_agent)
        diva_agent.getMotorDynamics()
        diva_agent.vocalize()
        simulation_data.appendData(diva_agent)
        print('Testing random model, vocalization: {}'.format(i))
        print(diva_agent.motor_command)

    
    g,ax2=initializeFigure();
    g,ax2=simulation_data.plotSimulatedData(g,ax2,'sensor', 0, 'sensor', 3,"or")
    
    plt.show();
    
