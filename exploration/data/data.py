"""
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
"""

from .DataTemplates.TabularData import TabularData
import matplotlib.pyplot as plt
from .PlotTools import movingAverage
import pandas as pd

class Object(object):
    def __init__(self):
        pass

class SimulationData():
    def __init__(self, system, prelocated_samples=100000):
        self.motor = TabularData(system.motor_names, prelocated_samples=prelocated_samples)
        self.sensor = TabularData(system.sensor_names, prelocated_samples=prelocated_samples)
        self.sensor_goal = TabularData(system.sensor_names, prelocated_samples=prelocated_samples)
        self.somato = TabularData(system.somato_names, prelocated_samples=prelocated_samples)
        self.somato_goal = TabularData(system.somato_names, prelocated_samples=prelocated_samples)
        self.competence = TabularData(['competence'], prelocated_samples=prelocated_samples)
        self.social = TabularData(system.sensor_names, prelocated_samples=prelocated_samples)
        self.cons = TabularData(['cons'], prelocated_samples = prelocated_samples)

    def append_data(self, system):
        self.motor.appendData(system.motor_command.flatten())
        self.sensor.appendData(system.sensor_out)
        self.sensor_goal.appendData(system.sensor_goal)
        self.somato.appendData(system.somato_out)
        self.somato_goal.appendData(system.somato_goal)
        self.competence.appendData(system.competence_result)
        self.social.appendData(system.sensor_instructor.flatten())
        self.cons.appendData(system.cons_out)

    def save_data(self, file_name):
        self.motor.data.to_hdf(file_name, 'motor')
        self.sensor.data.to_hdf(file_name, 'sensor')
        self.sensor_goal.data.to_hdf(file_name, 'sensor_goal')
        self.somato.data.to_hdf(file_name, 'somato')
        self.somato_goal.data.to_hdf(file_name, 'somato_goal')
        self.competence.data.to_hdf(file_name, 'competence')
        self.social.data.to_hdf(file_name, 'social')
        self.cons.data.to_hdf(file_name, 'cons')

    def cut_data(self, system, start, stop):
        simulationdata_tmp = SimulationData(system)
        simulationdata_tmp.motor.data = self.motor.data.iloc[start:stop]
        simulationdata_tmp.sensor.data = self.sensor.data.iloc[start:stop]
        simulationdata_tmp.sensor_goal.data = self.sensor_goal.data.iloc[start:stop]
        simulationdata_tmp.somato.data = self.somato.data.iloc[start:stop]
        simulationdata_tmp.somato_goal.data = self.somato_goal.data.iloc[start:stop]
        simulationdata_tmp.competence.data = self.competence.data.iloc[start:stop]
        simulationdata_tmp.social.data = self.social.data.iloc[start:stop]
        simulationdata_tmp.cons.data = self.cons.data.iloc[start:stop]
        return simulationdata_tmp

    def get_samples(self, system, idx):
        simulationdata_tmp = SimulationData(system)
        simulationdata_tmp.motor.data = self.motor.data.iloc[idx]
        simulationdata_tmp.sensor.data = self.sensor.data.iloc[idx]
        simulationdata_tmp.sensor_goal.data = self.sensor_goal.data.iloc[idx]
        simulationdata_tmp.somato.data = self.somato.data.iloc[idx]
        simulationdata_tmp.somato_goal.data = self.somato_goal.data.iloc[idx]
        simulationdata_tmp.competence.data = self.competence.data.iloc[idx]
        simulationdata_tmp.social.data = self.social.data.iloc[idx]
        simulationdata_tmp.cons.data = self.cons.data.iloc[idx]
        return simulationdata_tmp

    def cut_final_data(self):
        self.motor.data = self.motor.get_all()
        self.sensor.data = self.sensor.get_all()
        self.sensor_goal.data = self.sensor_goal.get_all()
        self.somato.data = self.somato.get_all()
        self.somato_goal.data = self.somato_goal.get_all()
        self.competence.data = self.competence.get_all()
        self.social.data = self.social.get_all()
        self.cons.data = self.cons.get_all()

    def mix_datasets(self, system, sim_2):
        sim_1 = SimulationData(system)
        sim_1.motor.data = self.motor.data.append(sim_2.motor.data)
        sim_1.sensor.data = self.sensor.data.append(sim_2.sensor.data)
        sim_1.sensor_goal.data = self.sensor_goal.data.append(sim_2.sensor.data)
        sim_1.somato.data = self.somato.data.append(sim_2.somato.data)
        sim_1.somato_goal.data = self.somato_goal.data.append(sim_2.somato_goal.data)
        sim_1.competence.data = self.competence.data.append(sim_2.competence.data)
        sim_1.cons.data = self.cons.data.append(sim_2.cons.data)
        return sim_1

    def copy(self, system):
        tmp = SimulationData(system)
        tmp.motor.data = self.motor.data.copy(deep=True)
        tmp.sensor.data = self.sensor.data.copy(deep=True)
        tmp.sensor_goal.data = self.sensor_goal.data.copy(deep=True)
        tmp.somato.data = self.somato.data.copy(deep=True)
        tmp.somato_goal.data = self.somato_goal.data.copy(deep=True)
        tmp.competence.data = self.competence.data.copy(deep=True)
        tmp.cons.data = self.cons.data.copy(deep=True)
        return tmp

    def plot_time_series(self, src, column, color='b', axes=None, moving_average=0):
        return plot_time_series_(self, src, column, color=color, moving_average=moving_average, axes=axes)

    def plot_2D(self, src1, column1, src2, column2, color='b', axes=None):
        return plot_2D_(self, src1, column1, src2, column2, color=color, axes=axes)


def load_sim_h5(file_name, system=None):
    motor = pd.read_hdf(file_name, 'motor')
    sensor = pd.read_hdf(file_name, 'sensor')
    sensor_goal = pd.read_hdf(file_name, 'sensor_goal')
    competence = pd.read_hdf(file_name, 'competence')

    is_social = True
    try:
        social = pd.read_hdf(file_name, 'social')
    except:
        is_social = False

    is_somato = True
    try:
        cons = pd.read_hdf(file_name, 'cons')
        somato = pd.read_hdf(file_name, 'somato')
        somato_goal = pd.read_hdf(file_name, 'somato_goal')
    except:
        cons = pd.read_hdf(file_name, 'somato')
        is_somato = False


    system = Object()
    system.motor_names = ['M' + str(x) for x in range(motor.shape[0])]
    system.sensor_names = ['S' + str(x) for x in range(sensor.shape[0])]

    if is_somato:
        system.somato_names = ['Som' + str(x) for x in range(somato.shape[0])]
    else:
        system.somato_names = ['SOM_FOO']

    tmp = SimulationData(system,prelocated_samples=1)

    tmp.motor.data = motor
    tmp.sensor.data = sensor
    tmp.sensor_goal.data = sensor_goal
    if is_somato:
        tmp.somato.data = somato
        tmp.somato_goal.data = somato_goal
    tmp.competence.data = competence
    tmp.cons.data = cons
    if is_social:
        tmp.social.data = social
    return tmp, system

def plot_2D_(data, src1, column1, src2, column2, color='b', axes=None):
    if axes is None:
        fig, axes = plt.subplots(1,1)
    plt.sca(axes)

    src1_ = getattr(data, src1)
    src2_ = getattr(data, src2)

    src1_name = list(src1_.data.columns.values)[column1]
    src2_name = list(src2_.data.columns.values)[column2]

    data1 = src1_.data[[src1_name]]
    data2 = src2_.data[[src2_name]]

    plt.plot(data1, data2, color)


def plot_time_series_(data, src, column, color='b', x=None, axes=None, moving_average=1):
    if moving_average==0: #To still support previous versions
        moving_average=1

    if axes is None:
        fig, axes = plt.subplots(1,1)
    plt.sca(axes)

    src_ = getattr(data, src)

    src_name = list(src_.data.columns.values)[column]

    data = src_.data[[src_name]]

    if x is None:
        x_data = range(len(data)-moving_average+1)
    else:
        x_data = x

    if moving_average > 1:
        try:
            data = movingAverage(data.as_matrix(), moving_average)
        except AttributeError:
            data = movingAverage(data, moving_average)

    plt.plot(x_data, data, color, lw=3)

SimulationData_v2 = SimulationData  #To be deleted... eventually
load_sim_h5_v2 = load_sim_h5  #To be deleted... eventually

"""
class SimulationData(object):
    def __init__(self, system, prelocated_samples = 100000):
        self.motor = TabularData(system.motor_names, prelocated_samples = prelocated_samples)
        self.sensor = TabularData(system.sensor_names, prelocated_samples = prelocated_samples)
        self.sensor_goal = TabularData(system.sensor_names, prelocated_samples = prelocated_samples)
        self.somato = TabularData(system.somato_names, prelocated_samples = prelocated_samples)
        self.competence = TabularData(['competence'], prelocated_samples = prelocated_samples)

    def append_data(self, system):
        self.motor.append_data(system.motor_command.flatten())
        self.sensor.append_data(system.sensor_out)
        self.sensor_goal.append_data(system.sensor_goal)
        self.somato.append_data(system.somato_out)
        self.competence.append_data(system.competence_result)

    def save_data(self, file_name):
        self.motor.data.to_hdf(file_name, 'motor')
        self.sensor.data.to_hdf(file_name, 'sensor')
        self.sensor_goal.data.to_hdf(file_name, 'sensor_goal')
        self.somato.data.to_hdf(file_name, 'somato')
        self.competence.data.to_hdf(file_name, 'competence')

    def cut_data(self, system, start, stop):
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

    def mix_datasets(self, system, sim_2):
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

    # def plot_time_series(self, fig, axes, src, column, color, moving_average=0):
    #     return plot_time_series_(self, fig, axes, src, column, color, moving_average)
    #
    # def plot_2D(self, fig, axes, src1, column1, src2, column2, color):
    #     return plot_2D_(self, fig, axes, src1, column1, src2, column2, color)


    def plot_time_series(self, src, column, color='b', axes=None, moving_average=0):
        return plot_time_series_(self, src, column, color=color, moving_average=moving_average, axes=axes)

    def plot_2D(self, src1, column1, src2, column2, color='b', axes=None):
        return plot_2D_(self, src1, column1, src2, column2, color=color, axes=axes)
        

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
"""

class SimulationDataSocial(SimulationData):
    def __init__(self, system, prelocated_samples= 100000):
        SimulationData.__init__(self, system, prelocated_samples=prelocated_samples)
        self.social = TabularData(system.sensor_names, prelocated_samples=prelocated_samples)

    def append_data(self, system):
        SimulationData.append_data(self, system)
        self.social.appendData(system.sensor_instructor.flatten())

    def save_data(self, file_name):
        SimulationData.save_data(self, file_name)
        self.social.data.to_hdf(file_name, 'social')

    def cut_data(self, system, start, stop):
        simulationdata_tmp = SimulationDataSocial(system)
        simulationdata_tmp.motor.data = self.motor.data.iloc[start:stop]
        simulationdata_tmp.sensor.data = self.sensor.data.iloc[start:stop]
        simulationdata_tmp.sensor_goal.data = self.sensor_goal.data.iloc[start:stop]
        simulationdata_tmp.somato.data = self.somato.data.iloc[start:stop]
        simulationdata_tmp.competence.data = self.competence.data.iloc[start:stop]
        simulationdata_tmp.social.data = self.social.data.iloc[start:stop]
        return simulationdata_tmp
