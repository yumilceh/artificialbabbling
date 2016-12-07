'''
Created on Sep, 2016

@author: Juan Manuel Acevedo Valle
'''
from Models.GeneralModels.Mixture import GMM as GMMmix

import numpy as np
from scipy import linalg as LA 

class ILGMM(GMMmix):
    '''
    classdocs
    '''

    def __init__(self, min_components):
        
        self.params={'init_components': min_components,
                     'max_step_components': 30,
                     'max_components':60,
                     'a_split': 0.8}
        GMMmix.__init__(self, min_components) 
        
          
    def train(self, data):
        if self.initialized:
            self.short_term_model = GMMmix(self.params['init_components'])
            self.short_term_model.getBestGMM(data, lims=[self.params['init_components'],self.params['max_step_components']])
            self.mergeGMM(self.merge_similar_gaussians(self.short_term_model.model))
                       
        else:
            self.getBestGMM(data, lims=[self.params['init_components'],self.params['max_step_components']])
            self.short_term_model = GMMmix(self.model.n_components)
            self.initialized = True
    
    def merge_similar_gaussians(self, gmm2):
        #Selecting high related Gaussians to be mixtured
        gmm1 = self.model
        similarity = get_similarity_matrix(gmm1, gmm2)
        similarity_tmp = similarity.flatten()
        total_similar = np.sum(similarity_tmp)
        
        similarity_tmp = (1/total_similar) * similarity_tmp
        
        changed_flag = False
        for i in np.arange(len(similarity_tmp)-1,-1,-1):
            if similarity_tmp[i]<=0.01:
                changed_flag = True
                indices = np.array(np.unravel_index(i,similarity.shape))
                gmm2 = self.mergeGMMComponents(gmm2, indices[0], indices[1])
        
        
        if changed_flag:
            return self.merge_similar_gaussians(gmm2) 
        else:
            return gmm2                  
                        
    def mergeGMM(self,gmm2):
        covars_ = self.model._get_covars() 
        means_ = self.model.means_
        weights_ = self.model.weights_ 
        
        covars_2 = gmm2._get_covars() 
        means_2 = gmm2.means_
        weights_2 = gmm2.weights_ 
        
        new_components = weights_2.shape[0]
        for i in range(new_components):
            covars_ = np.insert(covars_, [-1], covars_2[i], axis=0)
            means_ = np.insert(means_, [-1], means_2[i], axis=0)
            weights_ = np.insert(weights_, [-1], weights_2[i], axis=0)   
            
        self.model.covars_ = covars_
        self.model.means_ = means_
        self.model.weights_ = weights_
        self.model.n_components = self.model.n_components+new_components
    
    def mergeGMMComponents(self, gmm2, index1, index2):
        gauss1 = {'covariance': self.model._get_covars()[index1],
                  'mean': self.model.means_[index1],
                  'weight': self.model.weights_[index1]}
        gauss2 = {'covariance': gmm2._get_covars()[index2],
                  'mean': gmm2.means_[index2],
                  'weight': gmm2.weights_[index2]}
        gauss = merge_gaussians(gauss1, gauss2)
        
        covars_1 = self.model._get_covars() 
        means_1 = self.model.means_
        weights_1 = self.model.weights_ 
        
        covars_1[index1] = gauss['covariance']
        means_1[index1] = gauss['mean']
        weights_1[index1] = gauss['weight']
        
        covars_2 = gmm2._get_covars() 
        means_2 = gmm2.means_
        weights_2 = gmm2.weights_ 
        
        covars_2 = np.delete(covars_2, index2, 0)
        means_2 = np.delete(means_2, index2, 0)
        weights_2 = np.delete(weights_2, index2, 0)
        
        self.model.covars_ = covars_1
        self.model.means_ = means_1
        self.model.weights_ = weights_1
        
        gmm2.covars_ = covars_2
        gmm2.means_ = means_2
        gmm2.weights_ = weights_2
        gmm2.n_components = gmm2.n_components - 1
        
        return gmm2
        
        
    def mergeGaussians(self, index1, index2):
        gauss1 = {'covariance': self.model._get_covars()[index1],
                  'mean': self.model.means_[index1],
                  'weight': self.model.weights_[index1]}
        gauss2 = {'covariance': self.model._get_covars()[index2],
                  'mean': self.model.means_[index2],
                  'weight':  self.model.weights_[index2]}
        gauss = merge_gaussians(gauss1, gauss2)
        
        covars_ = self.model._get_covars() 
        means_ = self.model.means_
        weights_ = self.model.weights_ 
        
        covars_[index1] = gauss['covariance']
        means_[index1] = gauss['mean']
        weights_[index1] = gauss['weight']
        

        covars_ = np.delete(covars_, index2, 0)
        means_ = np.delete(means_, index2, 0)
        weights_ = np.delete(weights_, index2, 0)
        
        self.model.covars_ = covars_
        self.model.means_ = means_
        self.model.weights_ = weights_
        self.model.n_components = self.model.n_components-1
        
    def splitGaussian(self, index):
        gauss = {'covariance': self.model._get_covars()[index],
                  'mean': self.model.means_[index],
                  'weight': self.model.weights_[index]}
        
        gauss1, gauss2 = split_gaussian(gauss, self.params['a_split']) 
        
        covars_ = self.model._get_covars() 
        means_ = self.model.means_
        weights_ = self.model.weights_ 
        
        covars_[index] = gauss1['covariance']
        means_[index] = gauss1['mean']
        weights_[index] = gauss1['weight']
        
        covars_ = np.insert(covars_, [-1], gauss2['covariance'], axis=0)
        means_ = np.insert(means_, [-1], gauss2['mean'], axis=0)
        weights_ = np.insert(weights_, [-1], gauss2['weight'], axis=0)                
    
        self.model.covars_ = covars_
        self.model.means_ = means_
        self.model.weights_ = weights_
        self.model.n_components = self.model.n_components+1
        
def get_KL_divergence(gauss1,gauss2):
    detC1 = LA.det(gauss1['covariance'])
    detC2 = LA.det(gauss2['covariance'])
    logC2C1 = np.log(detC2 / detC1)
    
    invC2 = LA.inv(gauss2['covariance'])
    traceinvC2C1 = np.trace( np.dot(invC2, gauss1['covariance'] ))
    
    m2m1 =  np.matrix(gauss2['mean']-gauss1['mean'])
    invC1 = LA.inv(gauss1['covariance'])
    mTC1m = (m2m1)*invC1*np.transpose(m2m1)
    
    D = np.shape(gauss1['covariance'])[0]
    
    return logC2C1+traceinvC2C1+mTC1m-D     

def get_similarity_estimation(gauss1,gauss2):
    return (1.0/2.0)*(get_KL_divergence(gauss1, gauss2)+get_KL_divergence(gauss2,gauss1))                   

def get_similarity_matrix(gmm1, gmm2):
    n_comp_1 = gmm1.n_components  
    n_comp_2 = gmm2.n_components
    
    similarity_matrix=np.zeros((n_comp_1,n_comp_2)) 
    for i,(Mu, Sigma) in enumerate(zip(gmm1.means_, gmm1._get_covars())):
        gauss1 = {'covariance': Sigma, 'mean': Mu}
        for j,(Mu2, Sigma2) in enumerate(zip(gmm2.means_, gmm2._get_covars())):
            gauss2 = {'covariance': Sigma2, 'mean': Mu2}
            similarity_matrix[i,j] = get_similarity_estimation(gauss1, gauss2)
    
    return(similarity_matrix)

def merge_gaussians(gauss1,gauss2):
    weight1 = gauss1['weight']
    covar1 = gauss1['covariance']
    mean1 = gauss1['mean']
    
    weight2 = gauss2['weight']
    covar2 = gauss2['covariance']
    mean2 = gauss2['mean']    
    
    weight = weight1 + weight2
    
    f1 = weight1/weight
    f2 = weight2/weight
    
    mean = f1*mean1 + f2*mean2

    m1m2 = np.matrix(mean1 - mean2) 
    covar = f1*covar1 + f2*covar2 + f1*f2*np.transpose(m1m2)*m1m2

    return {'covariance': covar, 'mean': mean, 'weight': weight}

def split_gaussian(gauss, a):  #Supervise that all the values here are real
    weight = gauss['weight']
    covar = gauss['covariance']
    mean = gauss['mean']    
    
    w, v_ = LA.eig(covar)
    max_eig_index = np.argmax(w)
    
    l = w[max_eig_index]
    v = v_[:,max_eig_index]
    
    Delta_v = np.matrix(np.sqrt(a*l) * v)
    weight = weight/2.0
    mean1 = np.matrix(mean + Delta_v)
    mean2 = np.matrix(mean - Delta_v)
    covar = covar - np.transpose(Delta_v) * Delta_v
    
    return {'covariance': covar, 'mean': mean1, 'weight': weight}, {'covariance': covar, 'mean': mean2, 'weight': weight}
    

    
    