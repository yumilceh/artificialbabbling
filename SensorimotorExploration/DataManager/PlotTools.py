'''
Created on May 11, 2016

@author: Juan Manuel Acevedo Valle
'''
import matplotlib.pyplot as plt
import numpy as np
def initializeFigure():
    '''
    Factory to make configured axes (
    '''
    fig, ax = plt.subplots(1, 1) # or what ever layout you want
    ax.hold(True)
    return fig, ax     

def movingAverage(vector, n_samples):
    return np.convolve(vector[:,0], np.ones((n_samples,))/n_samples, mode='valid') #modes={'full', 'same', 'valid'}