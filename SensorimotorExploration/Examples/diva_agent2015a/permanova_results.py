'''
Created on Oct 11, 2016

@author: Juan Manuel Acevedo Valle
'''
import numpy as np

if __name__ == '__main__':
    p_file = 'permanova_result_p.npy'
    ts_file = 'permanova_result_ts.npy'
    p_values = np.load(p_file)
    ts_values = np.load(ts_file)
    
    print("This is P")
    print(np.average(p_values,axis=2))
    print("This is TS")
    print(np.average(ts_values, axis=2))
    