"""
Created on Mar 15, 2017

@author: Juan Manuel Acevedo Valle
"""
import os
import numpy as np
import itertools

from SensorimotorExploration.DataManager.SimulationData import load_sim_h5
from SensorimotorExploration.DataManager.PlotTools import *


def create_dict(groups_k):
    return {k[0] + '_' + k[1]: [] for k in groups_k}


if __name__ == '__main__':

    directory = 'experiment_5/'
    data_files = os.listdir(directory)

    # Group by:
    type_ops = ['proprio', 'simple']
    mode_ops = ['autonomous', 'social']

    groups_k = list(itertools.product(type_ops, mode_ops))

    means_s = create_dict(groups_k)
    means_av_s = create_dict(groups_k)

    means_w = create_dict(groups_k)
    means_av_w = create_dict(groups_k)

    p_coll_s = create_dict(groups_k)
    p_coll_av_s = create_dict(groups_k)

    p_coll_w = create_dict(groups_k)
    p_coll_av_w = create_dict(groups_k)

    inter = create_dict(groups_k)
    inter_av = create_dict(groups_k)

    error_ev = create_dict(groups_k)
    error_ev_av = create_dict(groups_k)

    for data_file in (d_f for d_f in data_files if 'sim_data.h5' in d_f):
        data_file = directory + data_file
        conf_file = data_file.replace('sim_data.h5', 'conf.txt')
        conf = {}
        with open(conf_file) as f:
            for line in f:
                line = line.replace('\n', '')
                (key, val) = line.split(': ')
                conf[key] = val

        try:
            whole_data = load_sim_h5(data_file)
            interaction_data = whole_data.social.data.as_matrix(columns=None)
            interactions = np.zeros((interaction_data.shape[0],))
            interactions[~np.isnan(interaction_data[:, 0])] = 1

            social_data = load_sim_h5(data_file.replace('sim_data.h5', 'eva_valset.h5'))

            s_error_ = np.linalg.norm(social_data.sensor_goal.data.as_matrix() -
                                      social_data.sensor.data.as_matrix(), axis=1)

            s_p_con_v = sum(social_data.somato.data.as_matrix()) /\
                        len(social_data.somato.data.as_matrix())

            w_error_ = np.linalg.norm(whole_data.sensor_goal.data.as_matrix() -
                                      whole_data.sensor.data.as_matrix(), axis=1)

            w_p_con_v = sum(whole_data.somato.data.as_matrix()) / len(whole_data.somato.data.as_matrix())

            eva_errors = []
            with open(data_file.replace('sim_data.h5', 'eval_error.txt'), 'r') as f:
                for line in f:
                    line.replace('\n','')
                    eva_errors_str = line.split(': ')
                    eva_errors += [float(eva_errors_str[1])]

            means_s[conf['type'] + '_' + conf['mode']] += [np.mean(s_error_)]
            means_w[conf['type'] + '_' + conf['mode']] += [np.mean(w_error_)]

            p_coll_s[conf['type'] + '_' + conf['mode']] += [np.mean(s_p_con_v)]
            p_coll_w[conf['type'] + '_' + conf['mode']] += [np.mean(w_p_con_v)]

            inter[conf['type'] + '_' + conf['mode']] += [np.sum(interactions)]

            error_ev[conf['type'] + '_' + conf['mode']] += [eva_errors]

        except IOError:
            pass

    # color = ['or', 'ob', 'ok', 'og']
    # fig1, axarr1 = plt.subplots(1, 2)
    # fig1.suptitle('Mean evaluation error')
    #
    # fig2, axarr2 = plt.subplots(1, 2)
    # fig2.suptitle('Percentage of constraints violations')

    # f, axarr

    legend = []
    for i, k in enumerate(groups_k):
        group = k[0] + '_' + k[1]
        legend += [group]

        means_av_s[group] = np.mean(np.array(means_s[group]))
        means_av_w[group] = np.mean(np.array(means_w[group]))

        p_coll_av_s[group] = np.mean(np.array(p_coll_s[group]))
        p_coll_av_w[group] = np.mean(np.array(p_coll_w[group]))

        inter_av[group] = np.mean(np.array(inter[group]))


        error_ev_av[group] = np.mean(np.array(error_ev[group]),axis=0)
        # total_evaluations = len(error_ev[group][0])
        # total_error_ev = np.zeros((total_evaluations,))
        # for sim in error_ev[group]:
        #     total_error = total_error_ev + np.array(sim).flatten()

        # axarr1[0].plot(means[group]['whole'], color[i])
        # plt.hold(True)
        # axarr1[0].set_title('Whole dataset')
        # plt.legend(legend)
        #
        # axarr1[1].plot(means[group]['social'], color[i])
        # plt.hold(True)
        # axarr1[1].set_title('Social dataset')
        # plt.legend(legend)
        #
        # axarr2[0].plot(p_const_v[group]['whole'], color[i])
        # plt.hold(True)
        # axarr2[0].set_title('Whole dataset')
        # plt.legend(legend)
        #
        # axarr2[1].plot(p_const_v[group]['social'], color[i])
        # plt.hold(True)
        # axarr2[1].set_title('Social dataset')
        # plt.legend(legend)

    plt.figure()
    plt.plot(error_ev_av['proprio_social'], linestyle='-', marker='o', color='b')
    plt.hold(True)
    plt.plot(error_ev_av['proprio_autonomous'],  linestyle='-', marker='o', color='r')
    plt.plot(error_ev_av['simple_social'],  linestyle='-', marker='o', color='g')
    plt.plot(error_ev_av['simple_autonomous'],  linestyle='-', marker='o', color='k')


    # plt.show(block=True)

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
        means_av[group]['social']['mean'] = np.mean(np.array(groups[group]['social']['mean']))
        means_av[group]['social']['max'] = np.mean(np.array(groups[group]['social']['max']))
        means_av[group]['social']['min'] = np.mean(np.array(groups[group]['social']['min']))

        means_av[group]['whole']['mean'] = np.mean(np.array(groups[group]['whole']['mean']))
        means_av[group]['whole']['max'] = np.mean(np.array(groups[group]['whole']['max']))
        means_av[group]['whole']['min'] = np.mean(np.array(groups[group]['whole']['min']))


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

    # OLD VERSION
    """
        type_ops = ['proprio', 'simple']
    mode_ops = ['autonomous', 'social']

    groups_k = list(itertools.product(type_ops, mode_ops))
    groups = {k[0] + '_' + k[1]: {'social': {'mean': [],
                                                     'max': [],
                                                     'min': []},
                                  'whole': {'mean': [],
                                                    'max': [],
                                                    'min': []}} for k in groups_k}

    means_av = {k[0] + '_' + k[1]: {'social': {'mean': [],
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

            s_error_ = np.linalg.norm(social_data.sensor_goal.data.as_matrix()[:9,:] -
                                    social_data.sensor.data.as_matrix()[:9,:], axis=1)

            w_error_ = np.linalg.norm(whole_data.sensor_goal.data.as_matrix()[:2100,:] -
                                      whole_data.sensor.data.as_matrix()[:2100,:], axis=1)

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
        means_av[group]['social']['mean'] = np.mean(np.array(groups[group]['social']['mean']))
        means_av[group]['social']['max'] = np.mean(np.array(groups[group]['social']['max']))
        means_av[group]['social']['min'] = np.mean(np.array(groups[group]['social']['min']))

        means_av[group]['whole']['mean'] = np.mean(np.array(groups[group]['whole']['mean']))
        means_av[group]['whole']['max'] = np.mean(np.array(groups[group]['whole']['max']))
        means_av[group]['whole']['min'] = np.mean(np.array(groups[group]['whole']['min']))


        axarr[0].plot(groups[group]['whole']['mean'],color[i])
        plt.hold(True)
        axarr[0].set_title('Whole dataset')
        plt.legend(legend)

        axarr[1].plot(groups[group]['social']['mean'], color[i])
        plt.hold(True)
        axarr[1].set_title('Social dataset')
        plt.legend(legend)

    plt.draw()

    """

    # OLD VERSION 2
    """
    def create_dict(groups_k):
        return {k[0] + '_' + k[1]: {'social': [],
                                    'whole': []} for k in groups_k}


    if __name__ == '__main__':

        directory = 'experiment_1_cmf/'
        data_files = os.listdir(directory)

        # Group by:
        type_ops = ['proprio', 'simple']
        mode_ops = ['autonomous', 'social']

        groups_k = list(itertools.product(type_ops, mode_ops))
        means = create_dict(groups_k)
        means_av = create_dict(groups_k)
        p_const_v = create_dict(groups_k)
        p_const_v_av = create_dict(groups_k)

        for data_file in (d_f for d_f in data_files if 'sim_data.h5' in d_f):
            data_file = directory + data_file
            conf_file = data_file.replace('sim_data.h5', 'conf.txt')
            conf = {}
            with open(conf_file) as f:
                for line in f:
                    line = line.replace('\n', '')
                    (key, val) = line.split(': ')
                    conf[key] = val

            try:
                data = load_sim_h5(data_file)
                interaction_data = data.social.data.as_matrix(columns=None)

                # datita[~np.isnan(datita[:, 0]), :]

                social_data = load_sim_h5(data_file.replace('sim_data.h5', 'social_eva_valset.h5'))
                whole_data = load_sim_h5(data_file.replace('sim_data.h5', 'whole_eva_valset.h5'))

                s_error_ = np.linalg.norm(social_data.sensor_goal.data.as_matrix()[:9, :] -
                                          social_data.sensor.data.as_matrix()[:9, :], axis=1)

                s_p_con_v = sum(social_data.somato.data.as_matrix()[:9]) / 9.

                w_error_ = np.linalg.norm(whole_data.sensor_goal.data.as_matrix()[:2100, :] -
                                          whole_data.sensor.data.as_matrix()[:2100, :], axis=1)

                w_p_con_v = sum(whole_data.somato.data.as_matrix()[:2100]) / 2100.

                means[conf['type'] + '_' + conf['mode']]['social'] += [np.mean(s_error_)]
                means[conf['type'] + '_' + conf['mode']]['whole'] += [np.mean(w_error_)]

                p_const_v[conf['type'] + '_' + conf['mode']]['social'] += [np.mean(s_p_con_v)]
                p_const_v[conf['type'] + '_' + conf['mode']]['whole'] += [np.mean(w_p_con_v)]


            except IOError:
                pass

        color = ['or', 'ob', 'ok', 'og']
        fig1, axarr1 = plt.subplots(1, 2)
        fig1.suptitle('Mean evaluation error')

        fig2, axarr2 = plt.subplots(1, 2)
        fig2.suptitle('Percentage of constraints violations')

        # f, axarr

        legend = []
        for i, k in enumerate(groups_k):
            group = k[0] + '_' + k[1]
            legend += [group]

            means_av[group]['social'] = np.mean(np.array(means[group]['social']))
            means_av[group]['whole'] = np.mean(np.array(means[group]['whole']))

            p_const_v_av[group]['social'] = np.mean(np.array(p_const_v[group]['social']))
            p_const_v_av[group]['whole'] = np.mean(np.array(p_const_v[group]['whole']))

            axarr1[0].plot(means[group]['whole'], color[i])
            plt.hold(True)
            axarr1[0].set_title('Whole dataset')
            plt.legend(legend)

            axarr1[1].plot(means[group]['social'], color[i])
            plt.hold(True)
            axarr1[1].set_title('Social dataset')
            plt.legend(legend)

            axarr2[0].plot(p_const_v[group]['whole'], color[i])
            plt.hold(True)
            axarr2[0].set_title('Whole dataset')
            plt.legend(legend)

            axarr2[1].plot(p_const_v[group]['social'], color[i])
            plt.hold(True)
            axarr2[1].set_title('Social dataset')
            plt.legend(legend)

        plt.show(block=True)
    """
