"""
Created on Mar 20, 2017

@author: Juan Manuel Acevedo Valle
"""
#  Adding the projects folder to the path##
import os, sys, random

#  Adding libraries##
from SensorimotorExploration.DataManager.SimulationData import load_sim_h5
from SensorimotorExploration.DataManager.PlotTools import *

from numpy import linspace
import numpy as np
import itertools
import datetime

directory = 'experiment_1/'

if __name__ == '__main__':
    data_files = os.listdir(directory)

    type_ops = ['proprio', 'simple']
    mode_ops = ['autonomous', 'social']

    groups_k = list(itertools.product(type_ops, mode_ops))
    groups = {k[0] + '-' + k[1]: {'social': {'mean': [],
                                             'max': [],
                                             'min': []},
                                  'whole': {'mean': [],
                                            'max': [],
                                            'min': []}} for k in groups_k}

    groups_av = {k[0] + '-' + k[1]: {'social': {'mean': [],
                                                'max': [],
                                                'min': []},
                                     'whole': {'mean': [],
                                               'max': [],
                                               'min': []}} for k in groups_k}

    for data_file in (d_f for d_f in data_files if 'sim_data.h5' in d_f):
        data_file = directory + data_file
        conf_file = data_file.replace('sim_data.h5', 'conf.txt')
        eval_file = data_file.replace('sim_data.h5', 'eva_valset.h5')
        conf = {}

        raw_data = load_sim_h5(data_file)


        with open(conf_file) as f:
            for line in f:
                line = line.replace('\n', '')
                (key, val) = line.split(': ')
                conf[key] = val

        try:
            social_data = load_sim_h5(eval_file)

            s_error_ = np.linalg.norm(social_data.sensor_goal.data.as_matrix()[:132,:] -
                                      social_data.sensor.data.as_matrix()[:132,:], axis=1)

            groups[conf['type'] + '-' + conf['mode']]['social']['mean'] += [np.mean(s_error_)]
            groups[conf['type'] + '-' + conf['mode']]['social']['max'] += [np.max(s_error_)]
            groups[conf['type'] + '-' + conf['mode']]['social']['min'] += [np.min(s_error_)]

            # f, axarr = plt.subplots(3, 1)
            # f.suptitle(conf['type'] + '-' + conf['mode'])
            # f, axarr[0] = social_data.plot_2D(f, axarr[0], 'sensor', 0, 'sensor', 1, '.b')
            # plt.hold(True)
            # f, axarr[0] = social_data.plot_2D(f, axarr[0], 'sensor_goal', 0, 'sensor_goal', 1, '.r')
            #
            # f, axarr[1] = social_data.plot_2D(f, axarr[1], 'sensor', 3, 'sensor', 4, '.b')
            # plt.hold(True)
            # f, axarr[1] = social_data.plot_2D(f, axarr[1], 'sensor_goal', 3, 'sensor_goal', 4, '.r')
            #
            raw_error_ = np.linalg.norm(raw_data.sensor_goal.data.as_matrix()[:20000,:] -
                                        raw_data.sensor.data.as_matrix()[:20000,:], axis=1)
            #
            # f, axarr[2] = raw_data.plot_time_series(f, axarr[2], 'competence', 0, 'b', moving_average=100)

            groups[conf['type'] + '-' + conf['mode']]['whole']['mean'] += [np.mean(raw_error_)]
            groups[conf['type'] + '-' + conf['mode']]['whole']['max'] += [np.max(raw_error_)]
            groups[conf['type'] + '-' + conf['mode']]['whole']['min'] += [np.min(raw_error_)]

        except IOError:
            pass



    color = ['or', 'ob', 'ok', 'og']
    f, axarr = plt.subplots(1, 1)
    f.suptitle('Mean evaluation error')

    legend = []
    for i, k in enumerate(groups_k):
        group = k[0] + '-' + k[1]
        legend += [group]
        groups_av[group]['social']['mean'] = np.mean(np.array(groups[group]['social']['mean']))
        groups_av[group]['social']['max'] = np.mean(np.array(groups[group]['social']['max']))
        groups_av[group]['social']['min'] = np.mean(np.array(groups[group]['social']['min']))

        groups_av[group]['whole']['mean'] = np.mean(np.array(groups[group]['whole']['mean']))
        groups_av[group]['whole']['max'] = np.mean(np.array(groups[group]['whole']['max']))
        groups_av[group]['whole']['min'] = np.mean(np.array(groups[group]['whole']['min']))

        axarr.plot(groups[group]['social']['mean'], color[i])
        plt.hold(True)
        axarr.set_title('Social dataset')
        plt.legend(legend)

    plt.show(block=True)
