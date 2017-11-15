'''
Created on Feb 5, 2016

@author: yumilceh
'''

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg 
if __name__ == '__main__':
   
    ## Adding libraries##
    from SensorimotorSystems.Parabola import ConstrainedParabolicArea as System
    from Algorithm.ModelEvaluation import  generateTestDatafromrawData
    from Algorithm.StorageDataFunctions import loadSimulationData_h5
    
    ## Simulation Parameters ##
    system=System()
    
    file_name = 'parabola_dataset_2.h5'
    data = loadSimulationData_h5(file_name, system)
    
    ## Coverting data to Matrix form ##


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
    validation_data_set.save_data('parabola_dataset_2.h5')
    
    try:
        str_opt = raw_input("Press [enter] to continue or press 'Y' to keep plots...   ")
        if str_opt == 'Y':
            plt.show()
    except SyntaxError:
        plt.show()
        pass