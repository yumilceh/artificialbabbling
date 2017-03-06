"""
Created on Feb 5, 2016

@author: yumilceh
"""
from numpy import linspace
from numpy import random as np_rnd

if __name__ == '__main__':
    #  Adding the projects folder to the path##
    import os, sys, random

    sys.path.append("../../")

    #  Adding libraries##
    from SensorimotorExploration.Systems.Parabola import ConstrainedParabolicArea as System
    from SensorimotorExploration.DataManager.PlotTools import *

    # Creating system
    agent = System()

    fig1, ax1 = initializeFigure()
    fig1, ax1 = agent.drawSystem(fig1, ax1)
    plt.xlabel('S1')
    plt.ylabel('S2')
    plt.show()
