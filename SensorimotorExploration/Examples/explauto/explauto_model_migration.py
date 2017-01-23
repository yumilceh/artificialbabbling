'''
Created on Jan 23, 2017

@author: Juan Manuel Acevedo Valle
'''
if __name__ == '__main__':
    import sys, os
    import numpy as np
    sys.path.append("../../")
    
    from Models.explauto_SM import explauto_SM as SM_Model
    from SensorimotorSystems.Parabola import ConstrainedParabolicArea as System
    from Algorithm.RndSensorimotorFunctions import get_random_motor_set
    
    from DataManager.SimulationData import SimulationData


    system = System()
    sm_model = SM_Model(system, "nearest_neighbor")
    
    simulation_data = SimulationData(system)
    
    for m in get_random_motor_set(system, 100):
        s = system.setMotorCommand(m)
        simulation_data.appendData(system)
        sm_model.train(simulation_data)
        
    s_g = np.array([4, 3.5])
    sm_model.getMotorCommand(system,sensor_goal = s_g)
    
    from matplotlib import pyplot as plt
    plt.plot(s_g, 'ob')
    plt.hold(True)
    plt.plot(system.sensorOutput, 'xr')
    
    plt.show()
    
    pass 
    
    
    