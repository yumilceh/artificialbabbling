'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from GeneralModels.Mixture import GMM
import pandas as pd
import numpy as np 
import sys

class GMM_IM(object):
    '''
    classdocs
    '''


    def __init__(self, Agent, n_gauss_components):
        '''
        Constructor
        '''
        self.size_data=2*Agent.n_sensor+1
        self.goal_size=Agent.n_sensor
        self.competence_index=2*Agent.n_sensor+1; #Considering that one column will be time
        self.time_index=0;
        self.sensor_names=Agent.sensor_names
        self.GMM=GMM(n_gauss_components)
        
        
    def train(self,simulation_data):
        train_data_tmp=pd.concat([simulation_data.sensor_goal_data.data, simulation_data.sensor_data.data, simulation_data.competence_data.data], axis=1)
        train_data_tmp.reindex()
        train_data_tmp=train_data_tmp.reset_index()
        self.GMM.train(train_data_tmp.as_matrix(columns=None))
        
    def get_interesting_goal(self):
        goal_size=self.goal_size
                                
        #selecting randomly a GM to draw a sample s_g
        y_g=None
        
        gmm=self.GMM.model

        covariances = self.sortInterestingGaussians();
        if type(covariances)!=type(None):
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
                cumulated_cov=cumulated_cov+COVARIANCE
                #------------------------------------------ print(cumulated_cov)
        
        else:
            greatest_cov=-sys.float_info.max
            competence_index=self.competence_index
            time_index=self.time_index
            for (Mean,Covar) in (zip(gmm.means_, gmm._get_covars())):
                if(Covar[time_index,competence_index]>greatest_cov):
                    greatest_cov=Covar[time_index,competence_index]#Why absolute value???
                    y_g_tmp = np.random.multivariate_normal(Mean,Covar, 1);
                    y_g=y_g_tmp[1:1+goal_size];
        return y_g
    
    def sortInterestingGaussians(self):
        
        gmm=self.GMM.model
        
        K_IM=gmm.n_components    
           
        n_interesting_gauss=0
        
        index_interesting_gauss=np.zeros((K_IM))
        
        cov_interesting_gauss=np.zeros((K_IM))
        
        competence_index=self.competence_index
        time_index=self.time_index
        
        for k,(Covar,Mean) in enumerate(zip(gmm._get_covars(),gmm.means_)):
            if(Covar[time_index,competence_index]>0):
                index_interesting_gauss[n_interesting_gauss]=k;
                cov_interesting_gauss[n_interesting_gauss]=Covar[competence_index,time_index]; #abs(sum(GMM_IM.Sigma(1:6,7,k)));    %based on covariance respect to time/ covariance respect to auditory goals7
                n_interesting_gauss=n_interesting_gauss+1;

        #normalizing
        if (n_interesting_gauss>=1):
            index_interesting_gauss=index_interesting_gauss[0:n_interesting_gauss]
            cov_interesting_gauss=cov_interesting_gauss[0:n_interesting_gauss]
            min_cov=cov_interesting_gauss[np.argmin(cov_interesting_gauss)];
            cov_interesting_gauss=cov_interesting_gauss-min_cov;   
            total_cov=np.sum(cov_interesting_gauss);
            cov_interesting_gauss=cov_interesting_gauss[0:n_interesting_gauss]*(1/total_cov);
            covariances=pd.DataFrame({'INDEX':index_interesting_gauss,'COVARIANCE':cov_interesting_gauss})
            covariances=covariances.sort(['COVARIANCE'])
        
        else:
            covariances=None
    
        return covariances