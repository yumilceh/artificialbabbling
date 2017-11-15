"""
Created on Mar 8, 2017

@author: Juan Manuel Acevedo Valle
"""
from exploration.data.PlotTools import *


def show_results(system, simulation, val_data):
    fig1, ax1 = initializeFigure()
    fig1.suptitle('All Sensory Results')
    fig1, ax1 = simulation.data.plot_2D(fig1, ax1, 'sensor', 0, 'sensor', 1, ".b")
    plt.hold(True)
    fig1, ax1 = simulation.data.plot_2D(fig1, ax1, 'sensor', 3, 'sensor', 4, ".r")
    ax1.legend(['First Perception Window', 'Second Perception Window'])
    plt.xlabel('F1')
    plt.ylabel('F2')
    ax1.relim()
    ax1.autoscale_view()

    fig2, ax2 = initializeFigure()
    fig2.suptitle('Evaluation Error Evolution')
    plt.plot(simulation.evaluation_error[1:], 'b')
    plt.hold(True)
    plt.xlabel('Evaluation training step')
    plt.ylabel('Mean error')

    fig3, ax3 = initializeFigure()
    fig3.suptitle('Validation')
    fig3, ax3 = val_data.plot_2D(fig3, ax3, 'sensor_goal', 0, 'sensor_goal', 1, "xr")
    plt.hold(True)
    fig3, ax3 = val_data.plot_2D(fig3, ax3, 'sensor_goal', 3, 'sensor_goal', 4, "xb")
    fig3, ax3 = val_data.plot_2D(fig3, ax3, 'sensor', 0, 'sensor', 1, ".b")
    fig3, ax3 = val_data.plot_2D(fig3, ax3, 'sensor', 3, 'sensor', 4, ".r")
    ax3.legend(['Goal_1stWin', 'Goal_2ndWin', 'Result_1stWin', 'Result_2ndWin'])
    plt.xlabel('F1')
    plt.ylabel('F2')
    ax1.relim()

    fig5, ax5 = initializeFigure()
    fig2.suptitle('Competence during Exploration')
    simulation.data.plot_time_series(fig5, ax5, 'competence', 0, 'b', moving_average=10)
    simulation.data.plot_time_series(fig5, ax5, 'competence', 0, 'r', moving_average=200)
    ax5.legend(['Competence (win=10)', 'Competence (win=200)'])

    fig6, ax6 = initializeFigure()
    fig6.suptitle('All Sensory Goals')
    fig6, ax6 = simulation.data.plot_2D(fig6, ax6, 'sensor_goal', 0, 'sensor_goal', 1, ".b")
    plt.hold(True)
    fig6, ax6 = simulation.data.plot_2D(fig6, ax6, 'sensor_goal', 3, 'sensor_goal', 4, ".r")
    ax6.legend(['First Perception Window', 'Second Perception Window'])
    plt.xlabel('F1')
    plt.ylabel('F2')
    ax6.relim()
    ax6.autoscale_view()

    plt.draw()
    plt.pause(0.001)

    plt.show(block=True)
