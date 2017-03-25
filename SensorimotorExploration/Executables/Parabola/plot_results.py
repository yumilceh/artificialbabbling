"""
Created on Mar 8, 2017

@author: Juan Manuel Acevedo Valle
"""
from SensorimotorExploration.DataManager.PlotTools import *


def show_results(system, simulation, val_data, proprio_val):
    fig1, ax1 = initializeFigure()
    fig1.suptitle('All Sensory Results')
    fig1, ax1 = simulation.data.plot_2D(fig1, ax1, 'sensor', 0, 'sensor', 1, ".b")
    ax1.relim()
    ax1.autoscale_view()

    fig2, ax2 = initializeFigure()
    fig2.suptitle('Evaluation Error Evolution')
    plt.plot(simulation.evaluation_error[1:], 'b')
    plt.hold(True)
    plt.xlabel('Evaluation training step')
    plt.ylabel('Mean error')

    fig3, ax3 = initializeFigure()
    fig3.suptitle('Validation: S1 vs S2')
    fig3, ax3 = system.drawSystem(fig3, ax3)
    plt.hold(True)
    fig3, ax3 = val_data.plot_2D(fig3, ax3, 'sensor_goal', 0, 'sensor_goal', 1, "xr")
    plt.hold(True)
    fig3, ax3 = val_data.plot_2D(fig3, ax3, 'sensor', 0, 'sensor', 1, ".b")
    ax3.legend(['Goal', 'Result'])

    fig4, ax4 = initializeFigure()
    fig4.suptitle('Proprioceptive Prediction Evaluation')
    fig4, ax4 = system.drawSystem(fig4, ax4)

    for x in proprio_val:
        plt.plot(x[0], x[1], x[2])

    fig5,ax5=initializeFigure()
    fig5.suptitle('Competence during Exploration')
    fig5, ax5 = simulation.data.plot_time_series(fig5, ax5, 'competence', 0, 'b', moving_average=0)
    plt.hold(True)
    fig5, ax5 = simulation.data.plot_time_series(fig5, ax5, 'competence', 0, 'r', moving_average=1000)
    ax5.legend(['Competence', 'Competence (win=20)'])

    fig6, ax6 = initializeFigure()
    fig6.suptitle('All Sensory Goal Results')
    fig6, ax6 = simulation.data.plot_2D(fig6, ax6, 'sensor_goal', 0, 'sensor_goal', 1, ".b")
    ax6.relim()
    ax6.autoscale_view()

    plt.draw()
    plt.pause(0.001)

    plt.show(block=True)
