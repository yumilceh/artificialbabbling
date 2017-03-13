'''
Created on Feb 5, 2016

@author: yumilceh
'''



if __name__ == '__main__':
    ## Adding the projects folder to the path##
    import os,sys,random
    sys.path.append(os.getcwd())
    from SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a
    from DataManager.SimulationData import SimulationData
    from Models.GMM_SM  import GMM_SM
    from Models.GMM_SS  import GMM_SS
    from Models.GMM_IM  import GMM_IM 
    from DataManager.PlotTools import *
    from Algorithm.RndSensorimotorFunctions import *
    
    
        
    diva_agent=Diva_Proprio2015a()
    
    n_vocalizations=100;
    
    simulation_data=SimulationData(diva_agent);
    
    n_motor_commands=diva_agent.n_motor;
    
    motor_commands=get_random_motor_set(diva_agent,n_vocalizations)
    
    gmm_sm=GMM_SM(diva_agent,28)
    gmm_ss=GMM_SS(diva_agent,28)
    gmm_im=GMM_IM(diva_agent,50)
    
    for i in range(n_vocalizations):
        diva_agent.set_action(motor_commands[i, :])
        diva_agent.getMotorDynamics()
        diva_agent.vocalize()
        simulation_data.appendData(diva_agent)
        print('Random initialization, vocalization: {}'.format(i))

    gmm_sm.train(simulation_data)

    f,ax=initializeFigure();
    f,ax=simulation_data.plotSimulatedData2D(f,ax,'sensor', 0, 'sensor', 3,"ob")
    
    
    n_random_examples=50
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
    #------------------------------------------- print(diva_agent.motor_command)
    
    for i in range(n_random_examples):
        diva_agent.sensor_goal=sensor_goals[i]
        gmm_sm.get_action(diva_agent)
        diva_agent.getMotorDynamics()
        diva_agent.vocalize()
        get_competence_Moulin2013(diva_agent)
        simulation_data.appendData(diva_agent)
        print('Testing random model, vocalization: {}'.format(i))
        #--------------------------------------- print(diva_agent.motor_command)

    
    g,ax2=initializeFigure();
    g,ax2=simulation_data.plotSimulatedData2D(g,ax2,'sensor', 0, 'sensor', 3,"or")
        
    gmm_im.train(simulation_data)
    n_chosen_experiments=50
    
    for i in range(n_chosen_experiments):
        diva_agent.sensor_goal=gmm_im.get_goal()
        gmm_sm.get_action(diva_agent)
        diva_agent.getMotorDynamics()
        diva_agent.vocalize()
        get_competence_Moulin2013(diva_agent)
        simulation_data.appendData(diva_agent)
        print('Testing interesting model, vocalization: {}'.format(i))
        #--------------------------------------- print(diva_agent.motor_command)
         
    h,ax3=initializeFigure();
    h,ax3=simulation_data.plotSimulatedData2D(h,ax3,'sensor', 0, 'sensor', 3,"or")
    
    j,ax4=initializeFigure();
    j,ax4=simulation_data.plotTemporalSimulatedData(j,ax4,'competence', 0,"r")
    
    plt.show();
    
    
    
