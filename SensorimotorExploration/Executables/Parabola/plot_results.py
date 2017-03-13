"""
Created on Mar 8, 2017

@author: Juan Manuel Acevedo Valle
"""
from SensorimotorExploration.DataManager.PlotTools import *


def show_results(system, simulation, val_data, proprio_val):
    fig1, ax1 = initializeFigure()
    fig1.suptitle('All Sensory Results')
    fig1, ax1 = simulation.data.plotSimulatedData2D(fig1, ax1, 'sensor', 0, 'sensor', 1, ".b")
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
    fig3, ax3 = val_data.plotSimulatedData2D(fig3, ax3, 'sensor_goal', 0, 'sensor_goal', 1, "xr")
    plt.hold(True)
    fig3, ax3 = val_data.plotSimulatedData2D(fig3, ax3, 'sensor', 0, 'sensor', 1, ".b")
    ax3.legend(['Goal', 'Result'])

    fig4, ax4 = initializeFigure()
    fig4.suptitle('Proprioceptive Prediction Evaluation')
    fig4, ax4 = system.drawSystem(fig4, ax4)

    for x in proprio_val:
        plt.plot(x[0], x[1], x[2])

    fig5,ax5=initializeFigure()
    fig5.suptitle('Competence during Exploration')
    simulation.data.plotTemporalSimulatedData(fig5,ax5,'competence',0,'b',moving_average=0)
    simulation.data.plotTemporalSimulatedData(fig5,ax5,'competence',0,'r',moving_average=20)
    ax5.legend(['Competence', 'Competence (win=20)'])

    fig6, ax6 = initializeFigure()
    fig6.suptitle('All Sensory Goal Results')
    fig6, ax6 = simulation.data.plotSimulatedData2D(fig6, ax6, 'sensor_goal', 0, 'sensor_goal', 1, ".b")
    ax6.relim()
    ax6.autoscale_view()

    plt.draw()
    plt.pause(0.001)

    plt.show(block=True)

    """
    fig2,ax2=initializeFigure()
    fig2.suptitle('Motor Commands: M1 vs M2')
    fig2,ax2=simulation_data.plotSimulatedData2D(fig2,ax2,'motor', 0, 'motor', 1,"or")
    fig2, ax2 = validation_valSet_data.plotSimulatedData2D(fig2,ax2,'motor', 0, 'motor', 1,"ob")
    fig2, ax2 = simulation1.models.f_sm.model.plotGMMProjection(fig2,ax2,0, 1)
    ax2.relim()
    ax2.autoscale_view()

    fig3,ax3=initializeFigure()
    fig3.suptitle('RESULTS: M1 vs S1')
    fig3,ax3=simulation_data.plotSimulatedData2D(fig3,ax3,'motor', 0, 'sensor', 0,"or")
    fig3, ax3 = validation_valSet_data.plotSimulatedData2D(fig3,ax3,'motor', 0, 'sensor', 0,"ob")
    fig3, ax3 = simulation1.models.f_sm.model.plotGMMProjection(fig3,ax3,0, 2)
    ax3.relim()
    ax3.autoscale_view()

    fig4,ax4=initializeFigure()
    fig4.suptitle('RESULTS: M2 vs S2')
    fig4,ax4=simulation_data.plotSimulatedData2D(fig4,ax4,'motor', 1, 'sensor', 1,"or")
    fig4, ax4 = validation_valSet_data.plotSimulatedData2D(fig4,ax4,'motor', 1, 'sensor', 1,"ob")
    fig4, ax4 = simulation1.models.f_sm.model.plotGMMProjection(fig4,ax4,1, 3)
    ax4.relim()
    ax4.autoscale_view()

    fig5,ax5=initializeFigure()
    fig5.suptitle('Initialization data: S1 vs S2')
    fig5,ax5=initialization_data_sm_ss.plotSimulatedData2D(fig5,ax5,'sensor', 0, 'sensor', 1,"or")
    fig5, ax5 = simulation1.models.f_sm.model.plotGMMProjection(fig5,ax5,2, 3)
    ax5.relim()
    ax5.autoscale_view()

    fig6, ax6=initializeFigure()
    fig6.suptitle('Inialization data: S_g1 vs S_g2')
    fig6, ax6=initialization_data_im.plotSimulatedData2D(fig6,ax6,'sensor_goal', 0, 'sensor_goal', 1,"ob")
    plt.hold(True)
    fig6, ax6=simulation_data.plotSimulatedData2D(fig6,ax6,'sensor_goal', 0, 'sensor_goal', 1,"or")
    fig6, ax = simulation1.models.f_im.model.plotGMMProjection(fig6,ax6,1, 2)
    ax6.relim()
    ax6.autoscale_view()


    fig2, ax7 =  initializeFigure();
    fig2.suptitle('Evaluation Error Evolution')
    plt.plot(simulation1.evaluation_error[1:],'b')
    plt.hold(True)
    plt.xlabel('Sensorimotor training step')
    plt.ylabel('Mean error')



    fig3, ax3 =  initializeFigure();
    fig3.suptitle('Validation: S1 vs S2')
    fig3, ax3 = validation_valSet_data.plotSimulatedData2D(fig3, ax3,'sensor', 0, 'sensor', 1,"ob")
    plt.hold(True)
    fig3, ax3 = validation_valSet_data.plotSimulatedData2D(fig3,ax3,'sensor_goal', 0, 'sensor_goal', 1,"or")
    ax3.legend(['Results','Goal'])

    fig9, ax9 = initializeFigure();
    fig9.suptitle('Competence during Training')
    fig9, ax9 = simulation_data.plotTemporalSimulatedData(fig9,ax9,'competence', 0,"r",moving_average=10)

    fig10, ax10 = initializeFigure();
    fig10.suptitle('Competence and Error during validation')
    fig10, ax10 = validation_valSet_data.plotTemporalSimulatedData(fig10,ax10,'competence', 0,"--b",moving_average=10)
    fig10, ax10 = validation_valSet_data.plotTemporalSimulatedData(fig10,ax10,'error', 0,"r",moving_average=10)






"""