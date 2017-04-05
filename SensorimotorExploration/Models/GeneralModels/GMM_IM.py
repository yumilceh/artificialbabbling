'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from Models.GeneralModels.Mixture import GMM
from sklearn import mixture 
import pandas as pd
import numpy as np 
import sys
from termcolor import colored
from copy import deepcopy

class PARAMS(object):
    def __init__(self):
        pass;
    
class GMM_IM(object):
    '''
    classdocs
    '''


    def __init__(self, agent, n_gauss_components, im_step=120, n_training_samples=4200):
        '''
        Constructor
        '''
        
        self.params=PARAMS()
        self.params.im_step = im_step
        self.params.n_training_samples = n_training_samples
        self.params.size_data = 2*agent.n_sensor+1
        self.params.goal_size = agent.n_sensor
        self.params.competence_index = 2*agent.n_sensor+1; #Considering that one column will be time
        self.params.time_index = 0;
        self.params.sensor_names = agent.sensor_names
        self.params.n_components=n_gauss_components
        
        
        self.model = GMM(n_gauss_components)
        self.model.Active=np.array([1]*n_gauss_components)
        
    def train(self,simulation_data):       
        self.model.get_best_gmm(self.get_train_data(simulation_data), lims=[1, self.params.n_components])
        self.model.initialized=True
        
    def get_train_data(self, simulation_data):
        n_training_samples = self.params.n_training_samples
        data_size=len(simulation_data.sensor_data.data.index)
        if data_size > n_training_samples:
            sensor_data = simulation_data.sensor_data.data[data_size-n_training_samples:]
            sensor_goal_data = simulation_data.sensor_goal_data.data[data_size-n_training_samples:]
            competence_data = simulation_data.competence_data.data[data_size-n_training_samples:]
            train_data_tmp = pd.concat([sensor_goal_data, sensor_data, competence_data],axis=1)
        else:
            train_data_tmp = pd.concat([simulation_data.sensor_goal_data.data, simulation_data.sensor_data.data, simulation_data.competence_data.data], axis=1)
        train_data_tmp.reindex()
        train_data_tmp = train_data_tmp.reset_index()
        train_data = train_data_tmp.as_matrix(columns=None)
        
        return train_data
        
        
    def get_interesting_goal(self,agent):
        
        goal_size=self.params.goal_size
                                
        #selecting randomly a GM to draw a sample s_g
        y_g=None
        
        gmm=self.model.model

        covariances, non_positive_covariances = self.sortInterestingGaussians();
        
        if non_positive_covariances!=True:
            gmm_covars=gmm._get_covars()
            gmm_means=gmm.means_

            #------------------- n_interesting_models=len(covariances.index)
            random=0.99999999999*np.random.random(1);
            
            #----------------------------------------------------- print(random)
            cumulated_cov=0;
            for (INDEX,COVARIANCE) in (zip(covariances['INDEX'],covariances['COVARIANCE'])):
                if ((random>=cumulated_cov) and (random<(cumulated_cov+COVARIANCE))):
                    selected_gauss=int(INDEX) 
                    y_g_tmp = np.random.multivariate_normal(gmm_means[selected_gauss,:], gmm_covars[selected_gauss], 1)
                    y_g=y_g_tmp[0,1:1+goal_size]
                    return boundSensorGoal(agent,y_g)           
                cumulated_cov=cumulated_cov+COVARIANCE
                #------------------------------------------ print(cumulated_cov)
        
        else:
            greatest_cov= -1.0 * sys.float_info.max
            competence_index=self.params.competence_index
            time_index=self.params.time_index
            for (Mean,Covar) in (zip(gmm.means_, gmm._get_covars())):
                if(Covar[time_index,competence_index]>greatest_cov):
                    greatest_cov=Covar[time_index,competence_index]#Why absolute value???
                    y_g_tmp = np.random.multivariate_normal(Mean,Covar, 1);
                    y_g=y_g_tmp[1:1+goal_size];
            return boundSensorGoal(agent,y_g)
        
    def get_interesting_goal_proprioception(self,agent,sm_model,ss_model):
        
        goal_size=self.params.goal_size
                                
        #selecting randomly a GM to draw a sample s_g
        y_g=None
        
        gmm=self.model.model

        covariances, non_positive_covariances = self.sortInterestingGaussians();
        
        if np.sum(self.model.Active)==0:
            non_positive_covariances=True # Forces the model to take the highest covariance 
        
        if non_positive_covariances!=True:
            gmm_covars=gmm._get_covars()
            gmm_means=gmm.means_

            #------------------- n_interesting_models=len(covariances.index)
            random=0.999999999999999999*np.random.random(1);
            
            #----------------------------------------------------- print(random)
            cumulated_cov=0;
            for (INDEX,COVARIANCE) in (zip(covariances['INDEX'],covariances['COVARIANCE'])):
                if ((random>=cumulated_cov) and (random<(cumulated_cov+COVARIANCE))):
                    if self.model.Active[int(INDEX)] == 0:
                        return self.get_interesting_goal(agent)
                    selected_gauss=int(INDEX) 
                    self.model.Active[selected_gauss]=0     
                    y_g_tmp = np.random.multivariate_normal(gmm_means[selected_gauss,:], gmm_covars[selected_gauss], 1)
                    y_g=y_g_tmp[0,1:1+goal_size]
                    motor_command_tmp=sm_model.get_action(agent, boundSensorGoal(agent, y_g))
                    if ss_model.predictProprioceptiveEffect(agent, motor_command_tmp)==0:
                        return boundSensorGoal(agent,y_g)
                    else:
                        return self.get_interesting_goal(agent)          
                cumulated_cov=cumulated_cov+COVARIANCE
                #------------------------------------------ print(cumulated_cov)
        
        else: #Returns and element that does not produce contact or has the maximum covariance
            greatest_cov= -1.0 * sys.float_info.max  
            competence_index=self.params.competence_index
            time_index=self.params.time_index
            for (Mean,Covar) in (zip(gmm.means_, gmm._get_covars())):
                if(Covar[time_index,competence_index]>greatest_cov):
                    greatest_cov=Covar[time_index,competence_index]#Why absolute value???
                    y_g_tmp = np.random.multivariate_normal(Mean,Covar, 1);
                    y_g = y_g_tmp[1:1+goal_size]
                    motor_command_tmp=sm_model.get_action(agent, boundSensorGoal(agent, y_g))
                    if ss_model.getProprioceptivePrediction(agent, motor_command_tmp)==0:
                        return boundSensorGoal(agent,y_g)
            return boundSensorGoal(agent,y_g)
         
    
    def get_interesting_goals(self,agent,n_goals=1):
        ''' This function returns as many goals as requested according to the interest model, including the gausian generators'''
        goal_size=self.params.goal_size
                                
        #selecting randomly a GM to draw a sample s_g
        y_g=np.zeros((n_goals,goal_size))
        y_g_indexes=np.zeros((n_goals,1))
        gmm=self.model.model

        covariances, non_positive_covariances = self.sortInterestingGaussians();
        if non_positive_covariances!=True:
            print(colored('Non-possitive covariances for the interest model','red'))
            
        gmm_covars=gmm._get_covars()
        gmm_means=gmm.means_

        #------------------- n_interesting_models=len(covariances.index)
        for k_sample in range(n_goals):
            random=0.99999999999*np.random.random(1);
            
            #----------------------------------------------------- print(random)
            cumulated_cov=0;
            for (INDEX,COVARIANCE) in (zip(covariances['INDEX'],covariances['COVARIANCE'])):
                if ((random>=cumulated_cov) and (random<(cumulated_cov+COVARIANCE))):
                    selected_gauss=int(INDEX)
                    self.model.Active[selected_gauss]=0      
                    y_g_tmp = np.random.multivariate_normal(gmm_means[selected_gauss,:], gmm_covars[selected_gauss], 1)
                    y_g[k_sample,:]=boundSensorGoal(agent,y_g_tmp[0,1:1+goal_size])
                    y_g_indexes[k_sample,0]=INDEX           
                cumulated_cov=cumulated_cov+COVARIANCE
                #------------------------------------------ print(cumulated_cov)
            
        return y_g, y_g_indexes
    
    def sortInterestingGaussians(self,active_comp=None):
        ''' If all covariances are negative flag must be turned on'''
        if active_comp == None:
            gmm=self.model.model
            K_IM=gmm.n_components
            
        n_interesting_gauss=0
        
        index_interesting_gauss=np.zeros((K_IM))
        
        cov_interesting_gauss=np.zeros((K_IM))
        
        competence_index=self.params.competence_index
        time_index=self.params.time_index
        
        non_positive_covariances=False;
        
        for k,(Covar,Mean) in enumerate(zip(gmm._get_covars(),gmm.means_)):
            if(Covar[time_index,competence_index]>0):
                index_interesting_gauss[n_interesting_gauss]=k;
                cov_interesting_gauss[n_interesting_gauss]=Covar[competence_index,time_index]; #abs(sum(GMM_IM.Sigma(1:6,7,k)));    %based on covariance respect to time/ covariance respect to auditory goals7
                n_interesting_gauss=n_interesting_gauss+1;

        #normalizing
        if (n_interesting_gauss>1):
            index_interesting_gauss=index_interesting_gauss[0:n_interesting_gauss]
            cov_interesting_gauss=cov_interesting_gauss[0:n_interesting_gauss]
            min_cov=cov_interesting_gauss[np.argmin(cov_interesting_gauss)];
            cov_interesting_gauss=cov_interesting_gauss-min_cov;   
            total_cov=np.sum(cov_interesting_gauss);
            cov_interesting_gauss=cov_interesting_gauss[0:n_interesting_gauss]*(1/total_cov);
            covariances=pd.DataFrame({'INDEX':index_interesting_gauss,'COVARIANCE':cov_interesting_gauss})
            covariances=covariances.sort(['COVARIANCE'])
            
        elif(n_interesting_gauss==1):
            index_interesting_gauss=index_interesting_gauss[0:n_interesting_gauss]
            cov_interesting_gauss=cov_interesting_gauss[0:n_interesting_gauss]
            min_cov=cov_interesting_gauss[np.argmin(cov_interesting_gauss)];

            total_cov=np.sum(cov_interesting_gauss);
            cov_interesting_gauss=cov_interesting_gauss[0:n_interesting_gauss]*(1/total_cov);
            covariances=pd.DataFrame({'INDEX':index_interesting_gauss,'COVARIANCE':cov_interesting_gauss})
            covariances=covariances.sort(['COVARIANCE'])
    
        else:
            non_positive_covariances=True;
            for k,(Covar,Mean) in enumerate(zip(gmm._get_covars(),gmm.means_)):
                index_interesting_gauss[n_interesting_gauss]=k;
                cov_interesting_gauss[n_interesting_gauss]=Covar[competence_index,time_index]; #abs(sum(GMM_IM.Sigma(1:6,7,k)));    %based on covariance respect to time/ covariance respect to auditory goals7
                n_interesting_gauss=n_interesting_gauss+1;
                
            index_interesting_gauss=index_interesting_gauss[0:n_interesting_gauss]
            cov_interesting_gauss=cov_interesting_gauss[0:n_interesting_gauss]
            min_cov=cov_interesting_gauss[np.argmin(cov_interesting_gauss)];
            cov_interesting_gauss=cov_interesting_gauss-min_cov;   
            total_cov=np.abs(np.sum(cov_interesting_gauss));
            cov_interesting_gauss=cov_interesting_gauss[0:n_interesting_gauss]*(1/total_cov);
            covariances=pd.DataFrame({'INDEX':index_interesting_gauss,'COVARIANCE':cov_interesting_gauss})
            covariances=covariances.sort(['COVARIANCE'])
     
        return covariances, non_positive_covariances
    
    def activateAllComponents(self):
        self.model.Active=np.array([1]*self.params.n_components)

    
def boundSensorGoal(agent,y_g):
    n_sensor=agent.n_sensor;
    min_sensor_values = agent.min_sensor_values;
    max_sensor_values = agent.max_sensor_values;
    for i in range(n_sensor):
        if (y_g[i] < min_sensor_values[i]):
            y_g[i] = min_sensor_values[i]
        elif (y_g[i] > max_sensor_values[i]):
            y_g[i] = max_sensor_values[i]
    return y_g