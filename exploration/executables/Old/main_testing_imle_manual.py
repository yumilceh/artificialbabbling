'''
Created on Feb 5, 2016

@author: yumilceh
'''
import sys, os
import numpy as np
import random
    
class Params(object):
    '''
        This class generates the structurure of paramaters as required by the IMLE library
    '''
    def __init__(self,agent):
        self.in_dims = range(agent.n_motor)
        self.n_dims=agent.n_motor+agent.n_sensor
        self.out_dims = range(agent.n_motor,self.n_dims)
        self.min = (np.append(agent.min_motor_values, agent.min_sensor_values, axis = 0))
        self.max = (np.append(agent.max_motor_values, agent.max_sensor_values, axis = 0))
        
def set_motor_command(agent, m):
    agent.motor_command=m;

if __name__ == '__main__':

    random.seed(1234)
    #Adding required paths
    print(os.getcwd())
    sys.path.append(os.getcwd()) 
    sys.path.append('/home/yumilceh/Documents/IMLE/build/lib/')
    
    from SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a
    from Models.GeneralModels.IMLE import IMLE
    from Algorithm.StorageDataFunctions import loadSimulationData
    from DataManager.SimulationData import SimulationData
    from Algorithm.CompetenceFunctions import get_competence_Moulin2013
    from Algorithm.StorageDataFunctions import saveSimulationData
    from DataManager.PlotTools import *
    
    #Creating agent
    agent = Diva_Proprio2015a();
       
    #Loading data
    simulation_results = loadSimulationData('simulation_data_1stAttempt.tar.gz', agent)
    simulation_data = simulation_results['simulation_data']
    
    #Initializing IMLE model
    params = Params(agent)
    fa_SM = IMLE(params, mode='explore')
    
    #Generating Training and Validation Sets
    ratio_samples_val=0.2;
    n_samples=len(simulation_data.motor_data.data)
    n_samples_val=np.ceil(ratio_samples_val*n_samples).astype(int)
    n_samples_train=n_samples-n_samples_val;
    random_samples_val=random.sample(xrange(0,n_samples),n_samples_val)
    random_samples_train=[index for index in range(0,n_samples) if index not in random_samples_val]

    
    #Training
    progress=1;
    for i in random_samples_train:
        print('Training with sample {current} of {total}'.format(current=progress, total=n_samples_train))
        x_ = simulation_data.motor_data.data.iloc[i].as_matrix()
        y_ = simulation_data.sensor_data.data.iloc[i].as_matrix()
        fa_SM.update(x_.astype(float),y_.astype(float))
        progress=progress+1;
        
    #------------------------------------------ #Validation against Training set
    #---------------------------- validation_trainSet_data=SimulationData(agent)
    #--------------------------------------------------------------- progress=1;
    #-------------------------------------------- for i in random_samples_train:
        # print('Testing using sample {current} of {total} in the training set'.format(current=progress, total=n_samples_train))
        #------------- y_ = simulation_data.sensor.data.iloc[i].as_matrix()
#------------------------------------------------------------------------------ 
        # set_motor_command(agent, fa_SM.infer(fa_SM.out_dims,fa_SM.in_dims,y_.astype(float)))
        #---------------------------------------------- agent.getMotorDynamics()
        #------------------------------------------- agent.execute_action()
        #-------------------------------------- get_competence_Moulin2013(agent)
        #---------------------------- validation_trainSet_data.append_data(agent)
        #-------------------------------------------------- progress=progress+1;
        
        
        
    #Validation against Validation set
    validation_valSet_data=SimulationData(agent)
    progress=1;
    for i in random_samples_val:
        print('Testing using sample {current} of {total} in the validation set'.format(current=progress, total=n_samples_val))
        y_ = simulation_data.sensor_data.data.iloc[i].as_matrix()
        
        set_motor_command(agent, fa_SM.infer(fa_SM.out_dims,fa_SM.in_dims,y_.astype(float)))
        agent.getMotorDynamics()
        agent.execute_action()
        get_competence_Moulin2013(agent)
        validation_valSet_data.append_data(agent)
        progress=progress+1;
        
    #---------- validation_trainSet_data.save_data('validation_trainSet_data.h5')
    validation_valSet_data.save_data('validation_valSet_data.h5')
        
    #---------------------------------------------- fig1,ax1=initializeFigure();
    # fig1,ax1=validation_trainSet_data.plot_time_series(fig1,ax1,'competence', 0,"r",moving_average=5000)
#------------------------------------------------------------------------------ 
    fig2,ax2=initializeFigure();
    fig2,ax2=validation_valSet_data.plot_time_series(fig2, ax2, 'competence', 0, "r", moving_average=5000)
    
    plt.show();