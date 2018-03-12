"""
Created on Feb 5, 2016

@author: yumilceh
"""

if __name__ == '__main__':
    #  Adding the projects folder to the path##
    import sys

    sys.path.append("../../")

    #  Adding libraries##
    from exploration.systems.trash.Parabola_Test import ParabolicRegion as System
    from exploration.data.PlotTools import *

    # Creating system
    agent = System()

    fig1, ax1 = initializeFigure()
    fig1, ax1 = agent.drawSystem(fig1, ax1)
    plt.xlabel('S1')
    plt.ylabel('S2')
    plt.show()
