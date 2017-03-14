"""
Created on May 23, 2016

@author: Juan Manuel Acevedo Valle
"""
import tarfile
import os
import pandas as pd
from ...DataManager.SimulationData import SimulationData
import h5py


def ndarray_to_h5(data, key_, file_name):
    with h5py.File(file_name, 'w') as hf:
        hf.create_dataset(key_, data=data)


def h5_to_ndarray(key_, file_name):
    with h5py.File(file_name, 'r') as hf:
        data = hf[key_][:]
    return data


def saveSimulationData(in_file_names, out_file_name):
    """ Saving Simulation into a tar.gz file"""
    tar = tarfile.open(out_file_name, "w:gz")
    for data_file in in_file_names:
        tar.add(data_file)
        os.remove(data_file)
    tar.close()


def loadSimulationData(file_name, agent):
    """Loading Simulation from a tar.gz file"""
    tar = tarfile.open(file_name, "r:gz")
    tar.extractall()
    simulation_data = dict()
    for partial_file_name in tar.getnames():
        var_name = partial_file_name[0:-3]
        tmp = SimulationData(agent)
        tmp.motor_data.data = pd.read_hdf(partial_file_name, 'motor_data')
        tmp.sensor_data.data = pd.read_hdf(partial_file_name, 'sensor_data')
        tmp.sensor_goal_data.data = pd.read_hdf(partial_file_name, 'sensor_goal_data')
        tmp.somato_data.data = pd.read_hdf(partial_file_name, 'somato_data')
        tmp.competence_data.data = pd.read_hdf(partial_file_name, 'competence_data')
        os.remove(partial_file_name)
        simulation_data[var_name] = tmp
    return simulation_data
