"""
Created on Jan 23, 2017

@author: Juan Manuel Acevedo Valle
"""
if __name__ == '__main__':
    import sys
    import numpy as np
    sys.path.append("../../")
    
    from SensorimotorExploration.Models.Sensorimotor.ExplautoSM import ExplautoSM as SM_Model
    from SensorimotorExploration.Systems.Parabola import ParabolicRegion as System
    from SensorimotorExploration.Algorithm.RndSensorimotorFunctions import get_random_motor_set
    from SensorimotorExploration.DataManager.SimulationData import SimulationData


    system = System()
    sm_model = SM_Model(system, "nearest_neighbor")
    sm_model.set_sigma_explo_ratio(0.01)
    
    simulation_data = SimulationData(system)
    
    for m in get_random_motor_set(system, 1000):
        s = system.set_action(m)
        system.executeMotorCommand()        
        simulation_data.appendData(system)
        sm_model.train(simulation_data)
        
    s_g = np.array([4, 3.5])
    
    sm_model.get_action(system, sensor_goal = s_g)
    
    # system.set_action(np.array([system.motor_command[1], system.motor_command[0]]))
    
    system.executeMotorCommand()
    
    from SensorimotorExploration.DataManager.PlotTools import initializeFigure, plt
    from matplotlib.pyplot import show
    
    fig1,ax1=initializeFigure()
    system.drawSystem(fig1,ax1)
    
    plt.plot(*s_g,  marker='o', color='blue')
    plt.hold(True)
    plt.plot(*system.sensorOutput,  marker='x', color='red')
        
    fig2,ax2=initializeFigure()
    simulation_data.plot_2D(fig2, ax2, 'sensor', 0, 'sensor', 1, 'or')
    
    show(block=True)

    pass
    
    
    