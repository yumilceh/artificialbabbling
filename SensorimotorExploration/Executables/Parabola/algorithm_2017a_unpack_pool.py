"""
Created on Mar 15, 2017

@author: Juan Manuel Acevedo Valle
"""

if __name__ == '__main__':
    #  Adding the projects folder to the path##
    import os, sys, random
    # sys.path.append("../../")



    #  Adding libraries##
    from SensorimotorExploration.DataManager.SimulationData import load_sim_h5
    from SensorimotorExploration.DataManager.PlotTools import *

    from numpy import linspace
    import numpy as np
    import itertools
    import datetime

    directory = 'experiment_10/'

    data_files = os.listdir(directory)

    type_ops = ['proprio', 'simple']
    mode_ops = ['autonomous', 'social']

    groups_k = list(itertools.product(type_ops, mode_ops))
    groups = {k[0] + '_' + k[1]: {'social': {'mean': [],
                                                     'max': [],
                                                     'min': []},
                                  'whole': {'mean': [],
                                                    'max': [],
                                                    'min': []}} for k in groups_k}

    groups_av = {k[0] + '_' + k[1]: {'social': {'mean': [],
                                                     'max': [],
                                                     'min': []},
                                  'whole': {'mean': [],
                                                    'max': [],
                                                    'min': []}} for k in groups_k}

    for data_file in (d_f for d_f in data_files if 'sim_data.h5' in d_f):
        data_file = directory + data_file
        conf_file = data_file.replace('sim_data.h5', 'conf.txt')
        conf = {}
        with open(conf_file) as f:
            for line in f:
                line = line.replace('\n','')
                (key, val) = line.split(': ')
                conf[key] = val

        try:
            social_data = load_sim_h5(data_file.replace('sim_data.h5', 'social_eva_valset.h5'))
            whole_data = load_sim_h5(data_file.replace('sim_data.h5', 'whole_eva_valset.h5'))

            s_error_ = np.linalg.norm(social_data.sensor_goal.data.as_matrix() -
                                    social_data.sensor.data.as_matrix(), axis=1)

            w_error_ = np.linalg.norm(whole_data.sensor_goal.data.as_matrix() -
                                      whole_data.sensor.data.as_matrix(), axis=1)

            groups[conf['type']+'_'+conf['mode']]['social']['mean'] += [np.mean(s_error_)]
            groups[conf['type'] + '_' + conf['mode']]['social']['max'] += [np.max(s_error_)]
            groups[conf['type'] + '_' + conf['mode']]['social']['min'] += [np.min(s_error_)]

            groups[conf['type']+'_'+conf['mode']]['whole']['mean'] += [np.mean(w_error_)]
            groups[conf['type'] + '_' + conf['mode']]['whole']['max'] += [np.max(w_error_)]
            groups[conf['type'] + '_' + conf['mode']]['whole']['min'] += [np.min(w_error_)]

        except IOError:
            pass

    color = ['or','ob','ok','og']
    f, axarr = plt.subplots(1, 2)
    f.suptitle('Mean evaluation error')

    legend=[]
    for i,k in enumerate(groups_k):
        group = k[0] + '_' + k[1]
        legend += [group]
        groups_av[group]['social']['mean'] = np.mean(np.array(groups[group]['social']['mean']))
        groups_av[group]['social']['max'] = np.mean(np.array(groups[group]['social']['max']))
        groups_av[group]['social']['min'] = np.mean(np.array(groups[group]['social']['min']))

        groups_av[group]['whole']['mean'] = np.mean(np.array(groups[group]['whole']['mean']))
        groups_av[group]['whole']['max'] = np.mean(np.array(groups[group]['whole']['max']))
        groups_av[group]['whole']['min'] = np.mean(np.array(groups[group]['whole']['min']))


        axarr[0].plot(groups[group]['whole']['mean'],color[i])
        plt.hold(True)
        axarr[0].set_title('Whole dataset')
        plt.legend(legend)

        axarr[1].plot(groups[group]['social']['mean'], color[i])
        plt.hold(True)
        axarr[1].set_title('Social dataset')
        plt.legend(legend)

    plt.draw()

    #  Plot min max and mins
    """
    #  Plot mean, max and mins
    color = ['.r','.b','.k','.g']
    f, axarr = plt.subplots(1, 3)
    f2, axarr2 = plt.subplots(1, 3)
    f.suptitle('Whole evaluation')
    f2.suptitle('Social evaluation')

    legend=[]
    for i,k in enumerate(groups_k):
        group = k[0] + '_' + k[1]
        legend += [group]
        groups_av[group]['social']['mean'] = np.mean(np.array(groups[group]['social']['mean']))
        groups_av[group]['social']['max'] = np.mean(np.array(groups[group]['social']['max']))
        groups_av[group]['social']['min'] = np.mean(np.array(groups[group]['social']['min']))

        groups_av[group]['whole']['mean'] = np.mean(np.array(groups[group]['whole']['mean']))
        groups_av[group]['whole']['max'] = np.mean(np.array(groups[group]['whole']['max']))
        groups_av[group]['whole']['min'] = np.mean(np.array(groups[group]['whole']['min']))


        axarr[0].plot(groups[group]['whole']['mean'],color[i])
        plt.hold(True)
        axarr[0].set_title('Mean')
        axarr[1].plot(groups[group]['whole']['max'], color[i])
        plt.hold(True)
        axarr[1].set_title('Max')
        axarr[2].plot(groups[group]['whole']['min'], color[i])
        plt.hold(True)
        axarr[2].set_title('Min')

        axarr2[0].plot(groups[group]['social']['mean'], color[i])
        plt.hold(True)
        axarr2[0].set_title('Mean')
        axarr2[1].plot(groups[group]['social']['max'], color[i])
        plt.hold(True)
        axarr2[1].set_title('Max')
        axarr2[2].plot(groups[group]['social']['min'], color[i])
        plt.hold(True)
        axarr2[2].set_title('Min')

    """