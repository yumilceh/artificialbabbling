'''
Created on Jan 23, 2017

@author: Juan Manuel Acevedo Valle
'''
if __name__ == '__main__':
    import sys, os
    import numpy as np
    
    from Models.explauto_IM import explauto_IM
    from SensorimotorSystems.Parabola import ConstrainedParabolicArea as System
    from Algorithm.CompetenceFunctions import get_competence_Moulin2013_explauto as competence


    system = System()
    model_type = 'discretized_progress'
    
    model = explauto_IM(system, model_type, competence)
    
    
#===============================================================================
#     sm_model = SM_Model(system, "nearest_neighbor")
#     sm_model.set_sigma_explo_ratio(0.01)
#     
#     simulation_data = SimulationData(system)
#     
#     for m in get_random_motor_set(system, 1000):
#         s = system.setMotorCommand(m)
#         system.executeMotorCommand()        
#         simulation_data.appendData(system)
#         sm_model.train(simulation_data)
#         
#     s_g = np.array([4, 3.5])
#     
#     sm_model.getMotorCommand(system,sensor_goal = s_g)
#     
#     # system.setMotorCommand(np.array([system.motor_command[1], system.motor_command[0]]))
#     
#     system.executeMotorCommand()
#     
#     from DataManager.PlotTools import * 
#     from matplotlib.pyplot import show
#     
#     fig1,ax1=initializeFigure()
#     system.drawSystem(fig1,ax1)
#     
#     plt.plot(*s_g,  marker='o', color='blue')
#     plt.hold(True)
#     plt.plot(*system.sensorOutput,  marker='x', color='red')
#         
#     fig2,ax2=initializeFigure()
#     simulation_data.plotSimulatedData2D(fig2,ax2,'sensor',0,'sensor',1,'or')
#     
#     show(block=True)
# 
#     pass
#     
#     
#     
#===============================================================================