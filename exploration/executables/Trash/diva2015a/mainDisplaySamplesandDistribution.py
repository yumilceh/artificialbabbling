'''
Created on Feb 5, 2016

@author: yumilceh
'''
import numpy as np

from exploration.DataVisualization.PlotTools import *

from exploration.data.data import SimulationData
from exploration.models.sensorimotor.trash.GMM_SM_ import GMM_SM
from exploration.systems.Diva2015a import DivaProprio2015a

if __name__ == '__main__':
    
    diva_agent=DivaProprio2015a()
    
    n_vocalizations=28;
    
    simulation_data=SimulationData(diva_agent);
    
    n_motor_commands=diva_agent.n_motor;
    
    motor_commands=np.random.random((n_vocalizations,n_motor_commands))
    
    gmm_sm=GMM_SM(diva_agent,28);
    
    for i in range(n_vocalizations):
        diva_agent.set_action(motor_commands[i, :])
        diva_agent.getMotorDynamics()
        diva_agent.vocalize()
        simulation_data.append_data(diva_agent)
        print(i)
     
    #===========================================================================
    # gmm_sm.train(simulation_data)
    #===========================================================================
    
    
    #======================================================8=====================
    # print(simulation_data.action.data)
    # print(simulation_data.sensor.data)
    #===========================================================================
    gmm_sm.train(simulation_data)
    

    #------------------------------------------ print(gmm_sm.GMM.model.weights_)
    #-------------------------------------------- print(gmm_sm.GMM.model.means_)
    #------------------------------------------- print(gmm_sm.GMM.model.covars_)
       
    f,ax=initializeFigure();
    
    f,ax=simulation_data.plotSimulatedData(f,ax,'sensor', 0, 'sensor', 3)
    
    f,ax=gmm_sm.GMM.plot_gmm_projection(f, ax, 0, 3)
    
    plt.show();          
    #===========================================================================
    # diva_agent.stop()
    # del(diva_agent)
    #===========================================================================
     
