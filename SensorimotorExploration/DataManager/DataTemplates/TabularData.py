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

    def __init__(self, varNames, prelocated_samples = 100000):
        '''
        Constructor
        '''
        nVars = len(varNames)
        self.nVars = nVars
        self.varNames = varNames
        zeros = np.zeros((prelocated_samples,nVars))
        self.data = pd.DataFrame(zeros, columns=varNames)
        self.current_idx = 0

    def appendData(self, new_data):
        single_sample = np.size(np.shape(new_data))
        if single_sample == 1 or single_sample == 0:
            self.data.iloc[self.current_idx] = new_data
            self.current_idx += 1


        else:
            n_new_samples = new_data.shape[0]
            #data_tmp = pd.DataFrame(new_data, columns=self.varNames)
            for i in range(n_new_samples):
                self.data.iloc[self.current_idx] = new_data[i,:]
                self.current_idx += 1

    def get_last(self, n):
        if self.current_idx is 0:
            raise ValueError
        return self.data.iloc[self.current_idx-n:self.current_idx]

    def get_all(self):
        if self.current_idx is 0:
            raise ValueError
        return self.data.iloc[:self.current_idx]

class TabularData_old(object):
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



            