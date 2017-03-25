"""
Created on May 26, 2016

@author: Juan Manuel Acevedo Valle
"""
import numpy as np
import itertools

def generate_motor_grid(system, n_samples):
    """ Currently works for 2D motor systems"""
    xmin = system.min_motor_values[0]
    xmax = system.max_motor_values[0]
    ymin = system.min_motor_values[1]
    ymax = system.max_motor_values[1]

    np_dim = np.ceil(np.sqrt(n_samples))

    m1, m2 = np.meshgrid(np.linspace(xmin,xmax,np_dim), np.linspace(ymin,ymax,np_dim))
    #grid = np.vstack([X.ravel(), Y.ravel()])
    return m1, m2

def get_random_motor_set(system, n_samples,
                         min_values=None,
                         max_values=None,
                         random_seed=np.random.randint(999, size=(1, 1))):
    """
        All vector inputs must be horizontal vectors
    """
    n_motor = system.n_motor

    raw_rnd_data = np.random.random((n_samples, n_motor))

    if min_values == None:
        min_values = system.min_motor_values
    if max_values == None:
        max_values = system.max_motor_values

    min_values = np.array(n_samples * [np.array(min_values)])
    max_values = np.array(n_samples * [np.array(max_values)])

    motor_commands = min_values + raw_rnd_data * (max_values - min_values)

    return motor_commands


def get_random_sensor_set(system, n_samples,
                          min_values=None,
                          max_values=None,
                          random_seed=np.random.randint(999, size=(1, 1))):
    """
        All vector inputs must be horizontal vectors
    """
    n_sensor = system.n_sensor

    raw_rnd_data = np.random.random((n_samples, n_sensor))

    if min_values == None:
        min_values = system.min_sensor_values
    if max_values == None:
        max_values = system.max_sensor_values

    min_values = np.array(n_samples * [np.array(min_values)])
    max_values = np.array(n_samples * [np.array(max_values)])

    sensor_commands = min_values + raw_rnd_data * (max_values - min_values)

    return sensor_commands

def get_table_from_dict(dict_):
    """This function is intended to convert a group of nested dictionaries into code to
        generate a the latex code toproduce a tables, the deep of final data must be the
        same for all branches"""
    dept, n_cols = get_dept(dict_)
    if dept is 0:
        return "Not a neasted dictionary."

    dict_ = {'dict': dict_}

    dept, n_cols = get_dept(dict_)

    latex_code = '\\begin{tabular}{|'
    for i in range(n_cols):
        latex_code += 'c|'
    latex_code += '}\n \\hline \n'


    sub_dict = dict_
    lines = [''] * dept
    for level in range(dept-1):
        dict_tmp = dict_
        mult_k_cols = n_cols

        key_levels = [dict_tmp.keys()]
        # n_k_cols = len(dict_tmp.keys())


        for i in range(level):
            dict_tmp = dict_tmp[dict_tmp.keys()[0]]
            key_levels += [dict_tmp.keys()]


        key_pool = itertools.product(*key_levels)



        for i,comb in enumerate(key_pool):
            dict_tmp = dict_
            for key_i in comb:
                dict_tmp = dict_tmp[key_i]
                n_k_cols = len(dict_tmp.keys())
                mult_k_cols /= n_k_cols

            lines[level] += get_dict_line(dict_tmp, mult_k_cols)
            mult_k_cols = n_cols

    for l in lines:
        li = l.rsplit('&', 1)
        l = ' '.join(li)
        latex_code += l + '\\\\ \n \\hline \n'

    latex_code += '\\end{tabular}'
    return latex_code

def get_dict_line(dict, mult_cols):
    latex_code = ''
    n_k_cols = len(dict.keys())
    for i, key in enumerate(dict.keys()):
        latex_code += '\\multicolumn{' + str(mult_cols) + '}{|c|}{' + key + '}'
        if i < n_k_cols - 1:
            latex_code += '&'
        else:
            latex_code += '&\n'
    return latex_code

def get_dept(dict={}):
    dept = 0
    n_columns = 1
    while True:
        try:
            keys = dict.keys()
        except AttributeError:
            """This  is the last level"""
            break

        if len(keys) is 0:
            """This  is the last level"""
            break
        else:
            dept += 1

        n_columns  = n_columns * len(dict.keys())
        dict = dict[keys[0]]

    return dept, n_columns
