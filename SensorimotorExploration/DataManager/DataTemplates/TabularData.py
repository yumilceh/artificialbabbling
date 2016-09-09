'''
Created on Feb 18, 2016

@author: Juan Manuel Acevedo Valle
'''
import pandas as pd
import numpy as np

class TabularData(object):
    '''
    classdocs
    '''


    def __init__(self, varNames):
        '''
        Constructor
        '''
        nVars=len(varNames)
        self.nVars=nVars
        self.varNames=varNames
        self.data=pd.DataFrame(columns=varNames)
        
    def appendData(self,newData):
        single_sample=np.size(np.shape(newData))
        if single_sample==1 or single_sample==0:
            data_tmp=pd.DataFrame([newData],columns=self.varNames)
            self.data=pd.concat([self.data, data_tmp],ignore_index=True);
            
        else:
            data_tmp=pd.DataFrame(newData,columns=self.varNames)
            self.data=pd.concat([self.data, data_tmp],ignore_index=True);
        
            