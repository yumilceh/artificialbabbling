"""
Created on Mar 13, 2017

@author: Juan Manuel Acevedo Valle
"""


def write_config_log(alg, file_name):
    with open(file_name, "a") as log_file:

        root_attr = ['name', 'mode', 'type']

        for attr_ in root_attr:
            if hasattr(alg, attr_):
                log_file.write(attr_ + ': ' + getattr(alg, attr_) + '\n')

        try:
            for key in alg.evaluation.data_file.keys():
                log_file.write(key + '_evaluation_file: ' + alg.evaluation.data_file[key] + '\n')
        except:
            pass

        for attr_ in dir(alg.params):
            if not attr_.startswith('__') and not callable(getattr(alg.params, attr_)):
                log_file.write('{}: {}\n'.format(attr_, getattr(alg.params, attr_)))

        root_attr_names = ['learner', 'instructor']
        for attr_ in root_attr_names:
            log = ''
            if hasattr(alg, attr_):
                try:
                    attr_log = getattr(alg, attr_).generate_log()
                    log += attr_ + ': {\n'
                    log += attr_log
                    log += '}\n'
                    log = log.replace('\n}', '}')
                except IndexError:
                    print("INDEX ERROR in Algorithm (root_attr_names) log generation")
                except AttributeError:
                    if isinstance(getattr(alg, attr_), dict):
                        log += attr_ + ': {\n'
                        for key in attr_.keys():
                            log += key + ': ' + str(attr_[key]) + ','
                        log += ('}\n')
                        log = log.replace(',}', '}')
                    else:
                        log += attr_ + ': ' + str(getattr(alg, attr_)) + '\n'
            log_file.write(log)

        root_object_names = ['models']
        for  attr_ in root_object_names:
            log_file.write('{')
            log_file.write(generate_object_log(getattr(alg, attr_))+'}\n')


def generate_object_log(OBJECT):
    log = ''
    for attr_ in dir(OBJECT):
        if attr_.startswith('__') or callable(getattr(OBJECT, attr_)):
            continue
        try:
            attr_log = getattr(OBJECT, attr_).generate_log()
            log += attr_ + ': {\n'
            log += attr_log
            log += '}\n'
            log = log.replace('\n}', '}')
        except IndexError:
            print("INDEX ERROR in Algorithm log generation")
        except AttributeError:
            if isinstance(getattr(OBJECT, attr_), dict):
                log += attr_ + ': {\n'
                for key in attr_.keys():
                    log += key + ': ' + str(attr_[key]) + ','
                log += ('}\n')
                log = log.replace(',}', '}')
            else:
                log += attr_ + ': ' + str(getattr(OBJECT, attr_)) + '\n'
    return log


def add_log_line(line, file_name):
    with open(file_name, "a") as log_file:
        log_file.write(line)


def read_config_log(file_name):
    conf = {}
    with open(file_name) as f:
        for line in f:
            try:
                line = line.replace('\n','')
                (key, val) = line.split(': ')
                conf[key] = val
            except:
                pass
    return conf