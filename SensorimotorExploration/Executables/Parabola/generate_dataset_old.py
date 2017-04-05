'''
Created on Feb 5, 2016

@author: yumilceh
'''
from numpy import linspace
from numpy import random as np_rnd
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
   
     
    ## Adding the projects folder to the path##
    import os,sys,random
    sys.path.append("../../")

    ## Adding libraries##
    from SensorimotorExploration.Systems.Parabola_Test import ParabolicRegion as System
    from SensorimotorExploration.Algorithm.AlgorithmRandom import Algorithm_Random as Algorithm
    from SensorimotorExploration.Algorithm.AlgorithmRandom import MODELS
    from SensorimotorExploration.Models.GMM_SM import GMM_SM
    from SensorimotorExploration.Models.GMM_SS import GMM_SS
    from SensorimotorExploration.DataManager.PlotTools import initializeFigure
   
    ## Simulation Parameters ##
    n_initialization=50
    n_experiments=10000
    
    random_seed=1234
    
    k_sm = 30
    sm_step=1000
    alpha_sm=0.05
    
    k_ss = 6
    ss_step=100
    alpha_ss=0.05

    ## To guarantee reproductible experiments##
    random.seed(random_seed)
    np_rnd.seed(random_seed)

    ## Creating Agent ##
    system=System()
    system.params.sigma_noise = 0.0000000001
    
    ## Creating Models ##
    models=MODELS()
    
    models.f_sm = GMM_SM(system,
                         k_sm,
                         sm_step=sm_step,
                         alpha=alpha_sm)
    models.f_ss = GMM_SS(system,
                         k_ss,
                         ss_step=ss_step,
                         alpha=alpha_ss)

    ## Creating Simulation object, running simulation and plotting experiments##
    file_prefix='Sinus_GMM_'
    simulation1=Algorithm(system,
                          models,
                          file_prefix=file_prefix,
                          n_experiments = n_experiments
                          )
    

    simulation1.runNonProprioceptiveAlgorithm()
    
    initialization_data_sm_ss=simulation1.data.initialization_data_sm_ss
    initialization_data_im=simulation1.data.initialization_data_im
    simulation_data=simulation1.data.simulation_data

    s_dims = 2
    data_as_matrix = simulation_data.sensor_data.data.as_matrix()
    somatodata_as_matrix =simulation_data.somato_data.data.as_matrix()
    
    fig1,ax1=initializeFigure();
    fig1.show()
    #===========================================================================
    # fig1.show()
    #===========================================================================
    fig1.canvas.draw()
    plt.scatter(data_as_matrix[:, 0], data_as_matrix[:, 1], 0.8, color='k')
    fig1.canvas.draw()
    plt.hold(True)    
    ax1.autoscale_view()
    
    data_as_matrix = data_as_matrix[np.where(somatodata_as_matrix[:,0] == 0)[:], :][0]
    plt.scatter(data_as_matrix[:, 0], data_as_matrix[:, 1], 0.8, color='r')
    fig1.canvas.draw()
    
    #===========================================================================
    # fig1,ax1=simulation_data.plot_2D(fig1,ax1,'sensor', 0, 'sensor', 1,"or")
    # fig1, ax1 = simulation1.models.f_sm.model.plot_gmm_projection(fig1,ax1,2, 3)
    #===========================================================================
    

    min_distance = 0.2
    change = True
    
    while change: #No optimal
        change = False
        for i in range(data_as_matrix.shape[0]):
            if change == True:
                break
            for j in range(i+1,data_as_matrix.shape[0]):
                distance_ij = np.linalg.norm(data_as_matrix[i,:]-data_as_matrix[j,:])
                if distance_ij < min_distance:
                    data_as_matrix = np.delete(data_as_matrix, j, 0)
                    change = True
                    break
                    
    plt.plot(data_as_matrix[:, 0], data_as_matrix[:, 1], "or")
    fig1.canvas.draw()
    
    import pandas as pd
    from DataManager.SimulationData import SimulationData
    validation_data_set = SimulationData(simulation1.agent)
    validation_data_set.sensor_data.data = pd.DataFrame(data_as_matrix)
    validation_data_set.saveData('parabola_dataset_2.h5')
    
    try:
        str_opt = raw_input("Press [enter] to continue or press 'Y' to keep plots...   ")
        if str_opt == 'Y':
            plt.show()
    except SyntaxError:
        plt.show()
        pass