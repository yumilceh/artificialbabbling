'''
Created on Nov 28, 2016

@author: Juan Manuel Acevedo Valle
'''
from sklearn.decomposition import IncrementalPCA

class DataIPCA(IncrementalPCA):
    '''
    This class retrieve a IPCA analysis of the requested data
    '''
    def __init__(self, n_components = None, batch_size = None,  params = None):
        '''
        Constructor
        '''       
        self = IncrementalPCA.__init__(n_components = n_components, batch_size = batch_size)
        self.params = params
              
        
        
        
    