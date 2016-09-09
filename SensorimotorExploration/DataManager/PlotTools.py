'''
Created on May 11, 2016

@author: Juan Manuel Acevedo Valle
'''
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def initializeFigure():
    '''
    Factory to make configured axes (
    '''
    fig, ax = plt.subplots(1, 1) # or what ever layout you want
    ax.hold(True)
    return fig, ax     

def initializeFigure3D():
    '''
    Factory to make configured axes (
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') # or what ever layout you want
    ax.hold(True)
    return fig, ax   

def movingAverage(vector, n_samples):
    return np.convolve(vector[:,0], np.ones((n_samples,))/n_samples, mode='valid') #modes={'full', 'same', 'valid'}