'''
Created on Feb 5, 2016

@author: yumilceh
'''
from numpy import linspace
from numpy import random as np_rnd

if __name__ == '__main__':
   
     
    ## Adding the projects folder to the path##
    import os,sys,random
    sys.path.append(os.getcwd())

    ## Adding libraries##
    from SensorimotorExploration.SensorimotorSystems.Parabola import ConstrainedParabolicArea as System
    from DataManager.PlotTools import * 
   
    ## Creating system
    agent = System()
    
    fig1,ax1=initializeFigure();
    fig1,ax1=agent.drawSystem(fig1,ax1)

    plt.show();