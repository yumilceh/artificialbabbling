"""
Created on Mar 8, 2017

@author: Juan Manuel Acevedo Valle
"""
from SensorimotorExploration.DataManager.PlotTools import *


def show_results(system, simulation, val_sm_data, val_ssm_data, proprio_val):
    fig1, ax1 = plt.subplots(1,1)
    fig1.suptitle('All Sensory Results')
    simulation.data.plot_2D('sensor', 0, 'sensor', 1, color=".b", axes=ax1)
    ax1.relim()
    ax1.autoscale_view()

    fig2, ax2 =  plt.subplots(1,1)
    fig2.suptitle('Evaluation Error Evolution')
    tmp = simulation.evaluation_error
    tmp = np.array(tmp)
    plt.plot(tmp[:-1,0],tmp[:-1,1], 'b') #last sample has i=-1 to force plot (Fix it!!!!!!)
    plt.hold(True)
    plt.xlabel('Evaluation training step')
    plt.ylabel('Mean error')

    fig3, ax3 =  plt.subplots(1,1)
    fig3.suptitle('Validation: Sensor1 vs Sensor2')
    plt.hold(True)
    val_sm_data.plot_2D('sensor_goal', 0, 'sensor_goal', 1, color="xr", axes=ax3)
    plt.hold(True)
    val_sm_data.plot_2D('sensor', 0, 'sensor', 1, color=".b", axes=ax3)
    ax3.legend(['Goal', 'Result'])

    fig4, ax4 =  plt.subplots(2,3)
    fig4.suptitle('Validation: Somato')

    subs=[]
    for i in range(4):
        for j in range(i+1,4):
            subs += [[i,j]]

    idx=0
    for i in range(2):
        for j in range(3):
            subs[idx]
            val_ssm_data.plot_2D('somato_goal', subs[idx][0], 'somato_goal', subs[idx][1], color="xr", axes=ax4[i, j])
            plt.hold(True)
            val_ssm_data.plot_2D('somato', subs[idx][0], 'somato', subs[idx][1], color=".b", axes=ax4[i, j])
            plt.xlabel(str(subs[idx][0]))
            plt.ylabel(str(subs[idx][1]))
            idx+=1

    # val_ssm_data.plot_2D('somato_goal', 0, 'somato_goal', 2, color="xr", axes=ax4[0, 1])
    # plt.hold(True)
    # val_ssm_data.plot_2D('somato', 0, 'somato', 2, color=".b", axes=ax4[0, 1])
    #
    #
    # val_ssm_data.plot_2D('somato_goal', 0, 'somato_goal', 3, color="xr", axes=ax4[0, 2])
    # plt.hold(True)
    # val_ssm_data.plot_2D('somato', 0, 'somato', 3, color=".b", axes=ax4[0, 2])
    #
    # val_ssm_data.plot_2D('somato_goal', 2, 'somato_goal', 3, color="xr", axes=ax4[1,1])
    # plt.hold(True)
    # val_ssm_data.plot_2D('somato', 2, 'somato', 3, color=".b", axes=ax4[1,1])
    # ax4[1,1].legend(['Goal', 'Result'])

    # fig4, ax4 =  plt.subplots(1,1)
    # fig4.suptitle('Proprioceptive Prediction Evaluation')
    # system.drawSystem(axes= ax4)
    #
    # for x in proprio_val:
    #     plt.plot(x[0], x[1], x[2])

    fig5,ax5 = plt.subplots(1,1)
    fig5.suptitle('Competence during Exploration')
    simulation.data.plot_time_series('competence', 0, color='b', moving_average=1, axes=ax5)
    plt.hold(True)
    simulation.data.plot_time_series('competence', 0, color='r', moving_average=100,axes=ax5)
    ax5.legend(['Competence', 'Competence (win=100)'])

    fig6, ax6 =  plt.subplots(1,1)
    fig6.suptitle('All Sensory Goal Results')
    simulation.data.plot_2D('sensor_goal', 0, 'sensor_goal', 1, color=".b",axes=ax6)
    ax6.relim()
    ax6.autoscale_view()

    plt.draw()
    plt.pause(0.001)

    plt.show(block=True)
