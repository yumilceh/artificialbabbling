'''
Created on Feb 16, 2016

@author: Juan Manuel Acevedo Valle
'''
from sklearn import mixture as mix

class GMM(object):
    '''
    classdocs
    '''

    def __init__(self, n_components):
        self.type='GMM'
        GMMtmp=mix.GMM(n_components=n_components,
                       covariance_type='diag',
                       random_state=None,
                       thresh=None, 
                       min_covar=0.001, 
                       n_iter=100, 
                       n_init=1, 
                       params='wmc', 
                       init_params='wmc' 
                       )
        self.model=GMMtmp;

    def train(self,data):
        print(data)
        self.model.fit(data)
        
        