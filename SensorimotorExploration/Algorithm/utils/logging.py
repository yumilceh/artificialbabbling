"""
Created on Mar 13, 2017

@author: Juan Manuel Acevedo Valle
"""


def write_config_log(alg, file_name):
    with open(file_name, "a") as log_file:

        root_attr = ['name', 'mode', 'type','f_sm_key',
                     'f_ss_key', 'f_im_key', 'f_cons_key']

        for attr_ in root_attr:
            if hasattr(alg, attr_):
                log_file.write(attr_ + ': ' + getattr(alg, attr_) + '\n')

        root_attr_names = ['learner', 'instructor']
        for attr_ in root_attr_names:
            if hasattr(alg, attr_):
                try:
                    log_file.write(attr_ + ': ' + getattr(getattr(alg, attr_), 'name') + '\n')
                    if attr_ is 'instructor':
                        try:
                            log_file.write(attr_ + ': ' + str(getattr(getattr(alg, attr_), 'idx_sensor')) + '\n')
                        except:
                            pass
                except IndexError:
                    pass #In case instructor is  None

        for attr_ in dir(alg.params):
            if not attr_.startswith('__') and not callable(getattr(alg.params, attr_)):
                log_file.write('{}: {}\n'.format(attr_, getattr(alg.params, attr_)))


def add_log_line(line, file_name):
    with open(file_name, "a") as log_file:
        log_file.write(line)


def read_config_log(file_name):
    conf = {}
    with open(file_name) as f:
        for line in f:
            line = line.replace('\n','')
            (key, val) = line.split(': ')
            conf[key] = val
    return conf