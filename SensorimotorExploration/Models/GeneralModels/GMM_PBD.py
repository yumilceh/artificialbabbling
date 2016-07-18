'''
Created on Jul 6, 2016

@author: Juan Manuel Acevedo Valle
'''
import os
import numpy as np
from Models.GeneralModels.Mixture import GMM as Mixture 
import pandas as pd
from scipy import linalg 


class PARAMS(object):
    def __init__(self):
        pass;
    
class MODEL(object):
    def __init__(self):
        pass;
    
    
class GMM(object):
    '''
    classdocs
    '''

    def __init__(self, n_components):
        self.params=PARAMS()
        self.files=PARAMS()
        self.model=MODEL()
        self.params.n_components=n_components
        self.params.model_id=str(np.random.randint(10000))
        self.initialized=False;
        print('Model ID: ' + str(self.params.model_id) + ' has been created.')
        
    def initializeMixture(self,data):
        self.getDataDimension(data)    
        GMM_tmp=Mixture(self.params.n_components)
        GMM_tmp.train(data)
        self.model.Priors = GMM_tmp.model.weights_
        self.model.Mu = GMM_tmp.model.means_
        self.model.Sigma = GMM_tmp.model._get_covars()
        self.generateFilesandFNames()
        saveModel(self)
        
    def train(self,data):  
        self.generateParamsAndDataFile(data)
        self.getDataDimension(data)
        self.generateFilesandFNames()
        command = "/home/yumilceh/Documents/pbdlib/build/examples/update_gmm "  #Automatize
        command = command + self.files.model_files_names + ' '
        command = command + self.files.data + ' '
        command = command + self.files.var_names  + ' '
        command = command + self.files.params     
        os.system( command )
        loadModel(self)
        
    def getRandomSamples(self, n_samples):
        GMM=Mixture(self.params.n_components)
        GMM.model.weights_=np.array(self.model.Priors)
        GMM.model.covars_=np.array(self.model.Sigma)
        GMM.model.means_=np.array(self.model.Mu)
        return GMM.model.sample(n_samples)
        
    def trainIncrementalLearning(self,new_data,alpha):
        #=======================================================================
        # if self.initialized:
        #     n_new_samples=np.size(new_data,0)
        #     n_persistent_samples=np.round(((1-alpha)*n_new_samples)/alpha)
        #     persistent_data=pd.DataFrame(self.getRandomSamples(n_persistent_samples),columns=new_data.columns)
        #     data=pd.concat([persistent_data,new_data],axis=0)
        #     self.train(data)
        # else:
        #=======================================================================
        self.initializeMixture(new_data)
        self.initialized=True
        self.train(new_data)
               
    def predict(self,x_dims, y_dims, y):
        '''
            This method returns the value of x that maximaze the probability P(x|y)
        '''
        y=np.mat(y)
        n_dimensions=np.amax(len(x_dims))+np.amax(len(y_dims))
        n_components=self.params.n_components
        gmm=self.model
        
        
        likely_x=np.mat(np.zeros((len(x_dims),n_components)))
        sm=np.mat(np.zeros((len(x_dims)+len(y_dims),n_components)))
        p_xy=np.mat(np.zeros((n_components,1)))
         
        for k,(Mu, Sigma) in enumerate(zip(gmm.Mu, gmm.Sigma)):
            Mu=np.transpose(Mu)
            #----------------------------------------------- Sigma=np.mat(Sigma)
            Sigma_yy=Sigma[:,y_dims]
            Sigma_yy=Sigma_yy[y_dims,:]
             
            Sigma_xy=Sigma[x_dims,:]
            Sigma_xy=Sigma_xy[:,y_dims]
            tmp1=linalg.inv(Sigma_yy)*np.transpose(y-Mu[y_dims])
            tmp2=np.transpose(Sigma_xy*tmp1)
            likely_x[:,k]=np.transpose(Mu[x_dims]+tmp2)
             
            #----------- sm[:,k]=np.concatenate((likely_x[:,k],np.transpose(y)))
            likely_x_tmp=pd.DataFrame(likely_x[:,k],index=x_dims)
            y_tmp=pd.DataFrame(np.transpose(y),index=y_dims)
            tmp3=pd.concat([y_tmp, likely_x_tmp])
            tmp3=tmp3.sort_index()
             
            sm[:,k]=tmp3.as_matrix()
             
            tmp4=1/(np.sqrt(((2.0*np.pi)**n_dimensions)*np.abs(linalg.det(Sigma))))
            tmp5=np.transpose(sm[:,k])-(Mu)
            tmp6=linalg.inv(Sigma)
            tmp7=np.exp((-1.0/2.0)*(tmp5*tmp6*np.transpose(tmp5))) #Multiply time GMM.Priors????
            p_xy[k,:]=np.reshape(tmp4*tmp7,(1))
            #- print('Warning: Priors are not be considering to compute P(x,y)')
             
        k_ok=np.argmax(p_xy);
        x=likely_x[:,k_ok];
         
        return np.array(x.transpose())[0]
        pass
    
        
    def generateFilesandFNames(self):
        self.files.var_names = 'var_names_' + self.params.model_id + '.txt'
        self.files.model_files_names = 'model_file_names_' + self.params.model_id + '.txt'
        self.files.data = 'data_file_train' + self.params.model_id + '.txt'   
        self.files.params = 'params_' + self.params.model_id + '.txt'
        
        self.files.Priors = 'GMM_priors_'  + self.params.model_id + '.txt'
        self.files.Sigma = 'GMM_sigma_'  + self.params.model_id + '.txt'
        self.files.Mu = 'GMM_mu_'  + self.params.model_id + '.txt'
        
        f = open(self.files.var_names, 'w')
        for i in range(len(self.params.var_names)-1):
            f.write(self.params.var_names[i] + ' ')
        f.write(self.params.var_names[-1])
        f.close()
        
        f = open(self.files.model_files_names, 'w')
        f.write(self.files.Priors + '\n')
        f.write(self.files.Mu + '\n')
        f.write(self.files.Sigma + '\n')
        f.write(self.files.var_names)
        f.close()
    
    def generateParamsAndDataFile(self, data):
        n_data = len(data.index)
        f = open(self.files.params, 'w')
        f.write(str(self.params.n_dims))
        f.write('\n')
        f.write(str(n_data))
        f.write('\n')
        f.write(str(self.params.n_components))
        f.close()
        np.savetxt( self.files.data, np.transpose(data.values), delimiter=' ',fmt='%f')
    
    def getDataDimension(self, data):
        n_dims = len(data.columns)
        var_names = []
        for i in range(n_dims):
            var_names.append('x' + str(i))
        self.params.var_names = var_names
        self.params.n_dims = n_dims    
            
    
    
def loadModel(GMM): 
    Priors = np.loadtxt(GMM.files.Priors)
    raw_Mu = np.loadtxt(GMM.files.Mu)
    raw_Sigma = np.loadtxt(GMM.files.Sigma)
    n_dims = GMM.params.n_dims
    n_components = GMM.params.n_components
    Mu = []
    Sigma = []
    for i in range(n_components):
        Mu.append(raw_Mu[:,i])
        Sigma.append(raw_Sigma[:,i*n_dims:(i+1)*n_dims])
        
    GMM.model.Priors=Priors;
    GMM.model.Mu=Mu;
    GMM.model.Sigma=Sigma;
    
def saveModel(GMM):
    n_dims = GMM.params.n_dims
    n_components = GMM.params.n_components
    raw_Mu = np.zeros((n_dims,n_components))
    raw_Sigma = np.zeros((n_dims,n_dims*n_components))
    raw_Priors=np.zeros((1,n_components))
    
    for k,(Mu, Sigma) in enumerate(zip(GMM.model.Mu, GMM.model.Sigma)):
        raw_Mu[:,k] = Mu;
        raw_Sigma[:,k*n_dims:(k+1)*n_dims] = Sigma
        raw_Priors[0,k] = GMM.model.Priors[k]
    np.savetxt(GMM.files.Priors,raw_Priors,fmt='%f')
    np.savetxt(GMM.files.Mu,raw_Mu,fmt='%f')
    np.savetxt(GMM.files.Sigma,raw_Sigma,fmt='%f')
    
    
    
    
    