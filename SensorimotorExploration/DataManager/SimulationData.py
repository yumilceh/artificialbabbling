"""
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
"""

from .DataTemplates.TabularData import TabularData
import matplotlib.pyplot as plt
from .PlotTools import movingAverage
import numpy as np
import pandas as pd


class Object(object):
    def __init__(self):
        pass


class SimulationData(object):
    """0,
    classdocs
    """

    def __init__(self, system, prelocated_samples = 100000):
        self.motor = TabularData(system.motor_names, prelocated_samples = prelocated_samples)
        self.sensor = TabularData(system.sensor_names, prelocated_samples = prelocated_samples)
        self.sensor_goal = TabularData(system.sensor_names, prelocated_samples = prelocated_samples)
        self.somato = TabularData(system.somato_names, prelocated_samples = prelocated_samples)
        self.competence = TabularData(['competence'], prelocated_samples = prelocated_samples)

    def appendData(self, system):
        self.motor.appendData(system.motor_command.flatten())
        self.sensor.appendData(system.sensor_out)
        self.sensor_goal.appendData(system.sensor_goal)
        self.somato.appendData(system.somato_out)
        self.competence.appendData(system.competence_result)

    def saveData(self, file_name):
        self.motor.data.to_hdf(file_name, 'motor')
        self.sensor.data.to_hdf(file_name, 'sensor')
        self.sensor_goal.data.to_hdf(file_name, 'sensor_goal')
        self.somato.data.to_hdf(file_name, 'somato')
        self.competence.data.to_hdf(file_name, 'competence')

    def cutData(self, system, start, stop):
        simulationdata_tmp = SimulationData(system)
        simulationdata_tmp.motor.data = self.motor.data.iloc[start:stop]
        simulationdata_tmp.sensor.data = self.sensor.data.iloc[start:stop]
        simulationdata_tmp.sensor_goal.data = self.sensor_goal.data.iloc[start:stop]
        simulationdata_tmp.somato.data = self.somato.data.iloc[start:stop]
        simulationdata_tmp.competence.data = self.competence.data.iloc[start:stop]
        return simulationdata_tmp

    def cut_final_data(self):
        self.motor.data = self.motor.get_all()
        self.sensor.data = self.sensor.get_all()
        self.sensor_goal.data = self.sensor_goal.get_all()
        self.somato.data = self.somato.get_all()
        self.competence.data = self.competence.get_all()

    def mixDataSets(self, system, sim_2):
        sim_1 = SimulationData(system)
        sim_1.motor.data = self.motor.data.append(sim_2.motor.data)
        sim_1.sensor.data = self.sensor.data.append(sim_2.sensor.data)
        sim_1.sensor_goal.data = self.sensor_goal.data.append(sim_2.sensor.data)
        sim_1.somato.data = self.somato.data.append(sim_2.somato.data)
        sim_1.competence.data = self.competence.data.append(sim_2.competence.data)
        return sim_1

    def copy(self, system):
        tmp = SimulationData(system)
        tmp.motor.data = self.motor.data.copy(deep=True)
        tmp.sensor.data = self.sensor.data.copy(deep=True)
        tmp.sensor_goal.data = self.sensor_goal.data.copy(deep=True)
        tmp.somato.data = self.somato.data.copy(deep=True)
        tmp.competence.data = self.competence.data.copy(deep=True)
        return tmp

    def plot_time_series(self, fig, axes, src, column, color, moving_average=0):
        return plot_time_series_(self, fig, axes, src, column, color, moving_average)

    def plot_2D(self, fig, axes, src1, column1, src2, column2, color):
        return plot_2D_(self, fig, axes, src1, column1, src2, column2, color)


class SimulationDataSocial(SimulationData):
    def __init__(self, system, prelocated_samples= 100000):
        SimulationData.__init__(self, system, prelocated_samples=prelocated_samples)
        self.social = TabularData(system.sensor_names, prelocated_samples=prelocated_samples)

    def appendData(self, system):
        SimulationData.appendData(self, system)
        self.social.appendData(system.sensor_instructor.flatten())

    def saveData(self, file_name):
        SimulationData.saveData(self, file_name)
        self.social.data.to_hdf(file_name, 'social')

    def cutData(self, system, start, stop):
        simulationdata_tmp = SimulationDataSocial(system)
        simulationdata_tmp.motor.data = self.motor.data.iloc[start:stop]
        simulationdata_tmp.sensor.data = self.sensor.data.iloc[start:stop]
        simulationdata_tmp.sensor_goal.data = self.sensor_goal.data.iloc[start:stop]
        simulationdata_tmp.somato.data = self.somato.data.iloc[start:stop]
        simulationdata_tmp.competence.data = self.competence.data.iloc[start:stop]
        simulationdata_tmp.social.data = self.social.data.iloc[start:stop]
        return simulationdata_tmp


class SimulationData_v2(object):
    def __init__(self, system, prelocated_samples=100000):
        self.motor = TabularData(system.motor_names, prelocated_samples=prelocated_samples)
        self.sensor = TabularData(system.sensor_names, prelocated_samples=prelocated_samples)
        self.sensor_goal = TabularData(system.sensor_names, prelocated_samples=prelocated_samples)
        self.somato = TabularData(system.somato_names, prelocated_samples=prelocated_samples)
        self.competence = TabularData(['competence'], prelocated_samples=prelocated_samples)
        self.social = TabularData(system.sensor_names, prelocated_samples=prelocated_samples)
        self.cons = TabularData(['cons'], prelocated_samples = prelocated_samples)

    def appendData(self, system):
        self.motor.appendData(system.motor_command.flatten())
        self.sensor.appendData(system.sensor_out)
        self.sensor_goal.appendData(system.sensor_goal)
        self.somato.appendData(system.somato_out)
        self.competence.appendData(system.competence_result)
        self.social.appendData(system.sensor_instructor.flatten())
        self.cons.appendData(system.cons_out)

    def saveData(self, file_name):
        self.motor.data.to_hdf(file_name, 'motor')
        self.sensor.data.to_hdf(file_name, 'sensor')
        self.sensor_goal.data.to_hdf(file_name, 'sensor_goal')
        self.somato.data.to_hdf(file_name, 'somato')
        self.competence.data.to_hdf(file_name, 'competence')
        self.social.data.to_hdf(file_name, 'social')
        self.cons.data.to_hdf(file_name, 'cons')

    def cutData(self, system, start, stop):
        simulationdata_tmp = SimulationData_v2(system)
        simulationdata_tmp.motor.data = self.motor.data.iloc[start:stop]
        simulationdata_tmp.sensor.data = self.sensor.data.iloc[start:stop]
        simulationdata_tmp.sensor_goal.data = self.sensor_goal.data.iloc[start:stop]
        simulationdata_tmp.somato.data = self.somato.data.iloc[start:stop]
        simulationdata_tmp.competence.data = self.competence.data.iloc[start:stop]
        simulationdata_tmp.social.data = self.social.data.iloc[start:stop]
        simulationdata_tmp.cons.data = self.cons.data.iloc[start:stop]
        return simulationdata_tmp

    def cut_final_data(self):
        self.motor.data = self.motor.get_all()
        self.sensor.data = self.sensor.get_all()
        self.sensor_goal.data = self.sensor_goal.get_all()
        self.somato.data = self.somato.get_all()
        self.competence.data = self.competence.get_all()
        self.social.data = self.social.get_all()
        self.cons.data = self.cons.get_all()

    def mixDataSets(self, system, sim_2):
        sim_1 = SimulationData(system)
        sim_1.motor.data = self.motor.data.append(sim_2.motor.data)
        sim_1.sensor.data = self.sensor.data.append(sim_2.sensor.data)
        sim_1.sensor_goal.data = self.sensor_goal.data.append(sim_2.sensor.data)
        sim_1.somato.data = self.somato.data.append(sim_2.somato.data)
        sim_1.competence.data = self.competence.data.append(sim_2.competence.data)
        sim_1.cons.data = self.cons.data.append(sim_2.cons.data)
        return sim_1

    def copy(self, system):
        tmp = SimulationData_v2(system)
        tmp.motor.data = self.motor.data.copy(deep=True)
        tmp.sensor.data = self.sensor.data.copy(deep=True)
        tmp.sensor_goal.data = self.sensor_goal.data.copy(deep=True)
        tmp.somato.data = self.somato.data.copy(deep=True)
        tmp.competence.data = self.competence.data.copy(deep=True)
        tmp.cons.data = self.cons.data.copy(deep=True)
        return tmp

    def plot_time_series(self, fig, axes, src, column, color, moving_average=0):
        return plot_time_series_(self, fig, axes, src, column, color, moving_average)

    def plot_2D(self, fig, axes, src1, column1, src2, column2, color):
        return plot_2D_(self, fig, axes, src1, column1, src2, column2, color)

def load_sim_h5_v2(file_name, system=None):
    motor = pd.read_hdf(file_name, 'motor')
    sensor = pd.read_hdf(file_name, 'sensor')
    sensor_goal = pd.read_hdf(file_name, 'sensor_goal')
    somato = pd.read_hdf(file_name, 'somato')
    competence = pd.read_hdf(file_name, 'competence')
    social = pd.read_hdf(file_name, 'social')
    cons = pd.read_hdf(file_name, 'cons')

    system = Object()
    system.motor_names = ['M' + str(x) for x in range(motor.shape[0])]
    system.sensor_names = ['S' + str(x) for x in range(sensor.shape[0])]
    system.somato_names = ['Som' + str(x) for x in range(somato.shape[0])]

    tmp = SimulationData_v2(system,prelocated_samples=1)

    tmp.motor.data = motor
    tmp.sensor.data = sensor
    tmp.sensor_goal.data = sensor_goal
    tmp.somato.data = somato
    tmp.competence.data = competence
    tmp.cons.data = cons
    tmp.social.data = social
    return tmp

def load_sim_h5(file_name, system=None):
    # Keeping support to old datamanager files
    try:
        motor = pd.read_hdf(file_name, 'motor')
        suff = ''
    except KeyError:
        motor = pd.read_hdf(file_name, 'motor_data')
        suff = '_data'

    sensor = pd.read_hdf(file_name, 'sensor' + suff)
    sensor_goal = pd.read_hdf(file_name, 'sensor_goal' + suff)
    somato = pd.read_hdf(file_name, 'somato' + suff)
    competence = pd.read_hdf(file_name, 'competence' + suff)
    try:
        social_ = pd.read_hdf(file_name, 'social' + suff)
        social = True
    except KeyError:
        social = False

    if system is None:
        system = Object()
        system.motor_names = ['M' + str(x) for x in range(motor.shape[0])]
        system.sensor_names = ['S' + str(x) for x in range(sensor.shape[0])]
        system.somato_names = ['Som' + str(x) for x in range(somato.shape[0])]

    if social:
        tmp = SimulationDataSocial(system,prelocated_samples=1)
        tmp.social.data = social_
    else:
        tmp = SimulationData(system,prelocated_samples=1)

    tmp.motor.data = motor
    tmp.sensor.data = sensor
    tmp.sensor_goal.data = sensor_goal
    tmp.somato.data = somato
    tmp.competence.data = competence
    return tmp


def plot_2D_(sim_data, fig, axes, src1, column1, src2, column2, color):
    motor_names = list(sim_data.motor.data.columns.values)
    sensor_names = list(sim_data.sensor.data.columns.values)
    somato_names = list(sim_data.somato.data.columns.values)
    if src1 == 'motor':
        x_name = motor_names[column1]
        data1 = sim_data.motor.data[[x_name]]
    elif src1 == 'sensor':
        x_name = sensor_names[column1]
        data1 = sim_data.sensor.data[[x_name]]
    elif src1 == 'somato':
        x_name = somato_names[column1]
        data1 = sim_data.somato.data[[x_name]]
    elif src1 == 'sensor_goal':
        x_name = sensor_names[column1]
        data1 = sim_data.sensor_goal.data[[x_name]]

    if src2 == 'motor':
        y_name = motor_names[column2]
        data2 = sim_data.motor.data[[y_name]]
    elif src2 == 'sensor':
        y_name = sensor_names[column2]
        data2 = sim_data.sensor.data[[y_name]]
    elif src2 == 'somato':
        y_name = somato_names[column2]
        data2 = sim_data.somato.data[[y_name]]
    elif src2 == 'sensor_goal':
        y_name = sensor_names[column2]
        data2 = sim_data.sensor_goal.data[[y_name]]

    plt.figure(fig.number)
    plt.sca(axes)
    plt.plot(data1, data2, color)
    return fig, axes


def plot_time_series_(sim, fig, axes, src, column, color, moving_average=0):
    motor_names = list(sim.motor.data.columns.values)
    sensor_names = list(sim.sensor.data.columns.values)
    somato_names = list(sim.somato.data.columns.values)
    if src == 'motor':
        x_name = motor_names[column]
        data = sim.motor.data[[x_name]]
    elif src == 'sensor':
        x_name = sensor_names[column]
        data = sim.sensor.data[[x_name]]
    elif src == 'somato':
        x_name = somato_names[column]
        data = sim.somato.data[[x_name]]
    elif src == 'competence':
        data = sim.competence.data[['competence']]
    elif src == 'sensor_goal':
        x_name = sensor_names[column]
        data = sim.sensor_goal.data[[x_name]]
    elif src == 'error':
        data = np.linalg.norm(sim.sensor_goal.data - sim.sensor.data, axis=1)
    elif src == 'error_log':
        data = np.log(sim.competence.data[['competence']])
    elif src is 'social':
        data = np.log(sim.social.data[['social']])

    if moving_average > 0:
        try:
            data = movingAverage(data.as_matrix(), moving_average)
        except AttributeError:
            data = movingAverage(data, moving_average)

    plt.figure(fig.number)
    plt.sca(axes)
    plt.plot(data, color)

    return fig, axes
