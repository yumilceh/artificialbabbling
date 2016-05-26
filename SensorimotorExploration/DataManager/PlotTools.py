'''
Created on May 11, 2016

@author: Juan Manuel Acevedo Valle
'''
import matplotlib.pyplot as plt

def initializeFigure():
    '''
    Factory to make configured axes (
    '''
    fig, ax = plt.subplots(1, 1) # or what ever layout you want
    ax.hold(True)
    return fig, ax     

def movingaverage():
    pass