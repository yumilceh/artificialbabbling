"""
Created on Mar 20, 2017

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

    directory = 'RndExperiments/'

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
            social_data = load_sim_h5(data_file.replace('sim_data.h5', 'eva_valset.h5'))

            s_error_ = np.linalg.norm(social_data.sensor_goal.data.as_matrix() -
                                    social_data.sensor.data.as_matrix(), axis=1)


            groups[conf['type']+'_'+conf['mode']]['social']['mean'] += [np.mean(s_error_)]
            groups[conf['type'] + '_' + conf['mode']]['social']['max'] += [np.max(s_error_)]
            groups[conf['type'] + '_' + conf['mode']]['social']['min'] += [np.min(s_error_)]

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

        axarr[1].plot(groups[group]['social']['mean'], color[i])
        plt.hold(True)
        axarr[1].set_title('Social dataset')
        plt.legend(legend)

    plt.draw()
