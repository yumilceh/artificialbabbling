"""
Created on Mar 8, 2017

@author: Juan Manuel Acevedo Valle
"""

import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    #  Adding the projects folder to the path##

    from SensorimotorExploration.Algorithm.utils.competence_funcs import exp_norm_error, exp_norm_moderate_error, exp_norm_error_exp
    err_min = -12.
    err_max = 12.

    err_med =  (err_max-err_min) / 2.
    n_samples = 100

    toy_error  = np.linspace(err_min,err_max,n_samples)

    MF_comp = [exp_norm_error(err) for err in list(toy_error)]
    Bar_comp = [exp_norm_moderate_error(err) for err in list(toy_error)]
    mix_comp = [exp_norm_error_exp(err) for err in list(toy_error)]

    plt.plot(toy_error, MF_comp, 'b')
    plt.hold(True)
    plt.plot(toy_error, Bar_comp, 'r')
    plt.plot(toy_error, mix_comp, 'k')

    plt.show()