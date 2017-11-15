"""
Created on Feb 16, 2017

@author: Juan Manuel Acevedo Valle
"""

import sys, datetime
import numpy as np

'''
#=======================================================================================
#
# ORIGINAL EXPERIMENT: Script to generate the recheable space of vowels using the 
# diva vocal tract 
#
#=======================================================================================
'''
explore = False  # Explore  for dataset

explore_vowels  =  False

report = False

file_conv = False

data_set_vowels = True

now = datetime.datetime.now().strftime("DSG_%Y_%m_%d_%H_%M_") # Data set Goal Space

file_name = "exploration_" + now

english_vowels = {'i': [296.0, 2241.0, 1.0], 'I': [396.0, 1839.0, 1.0], 'e': [532.0, 1656.0, 1.0],
                  'ae': [667.0, 1565.0, 1.0], 'A': [661.0, 1296.0, 1.0], 'a': [680.0, 1193.0, 1.0],
                  'b': [643.0, 1019.0, 1.0], 'c': [480.0, 857.0, 1.0], 'U': [395.0, 1408.0, 1.0],
                  'u': [386.0, 1587.0, 1.0], 'E': [519.0, 1408.0, 1.0]}

diva_output_scale = [100.0, 500.0, 1500.0, 3000.0]



if __name__ == '__main__' and explore:
    from exploration.systems.DivaStatic import DivaStatic
    from exploration.algorithm.DatasetGenerator import DatasetGenerator as Generator

    n_experiments = 1000000
    min_dist = 0.5
    n_save = 50000  # Save data each n_save samples

    system = DivaStatic()

    generator = Generator(system,
                          min_dist=min_dist,
                          n_experiments=n_experiments,
                          random_seed=np.random.random((1, 1)),
                          n_save_data=n_save,
                          file_prefix=file_name)

    generator.generete_dataset()
    reduced_data = generator.reduce_dataset()
    reduced_data.save_data('reduced_' + file_name)



if __name__ == '__main__' and explore_vowels:

    '''
     ORIGINAL EXPERIMENT: Script to look for art configurations producing english
     vowels using a random approach.
    '''
    from exploration.systems.DivaStatic import DivaStatic
    from exploration.algorithm.utils.functions import get_random_motor_set
    from exploration.data.data import SimulationData

    '''If normalization is needed these are the DivaMatlab scales,
    important to mention that for DivaStatic format frequencies
    are read UNORMALIZED!!!!'''

    english_vowels_keys = english_vowels.keys()

    distance_to_vowel = {key: None for key in english_vowels_keys}
    articulation_dict = {key: None for key in english_vowels_keys}

    n_experiments = 1000000
    n_save = 50000  # Save data each n_save samples

    system = DivaStatic()

    for vowel in english_vowels_keys:
        distance_to_vowel[vowel] = []
        articulation_dict[vowel] = SimulationData(system)

    # simulation_data = SimulationData(system)

    min_motor_values = np.array([-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 0.2, 0.2, 0.2])
    max_motor_values = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1])

    motor_commands = get_random_motor_set(system, n_experiments, min_values=min_motor_values,
                                          max_values=max_motor_values)

    for i in range(n_experiments):
        system.setMotorCommand(motor_commands[i, :])
        system.executeMotorCommand()

        # simulation_data.append_data(system)
        auditory_output = system.sensor_out

        if auditory_output[1] > 0.00001:
            for vowel in english_vowels_keys:
                ref_formants = english_vowels[vowel]

                F1 = auditory_output[1]
                F2 = auditory_output[2]
                distance = np.sqrt((F1 - ref_formants[0]) ** 2 + (F2 - ref_formants[1]) ** 2)
                if distance < distance_to_vowel[vowel][-1]:
                    distance_to_vowel[vowel] += [distance]
                    articulation_dict[vowel].append_data(system)

        print('Looking for vowels. Experiment: {} of {}'.format(i + 1, n_experiments))

        if np.mod(i, n_save) == 0:
            # simulation_data.save_data(file_name + '.h5')

            np.save(file_name + '_dist_to_vowel.npy', distance_to_vowel)
            np.save(file_name + '_articulations.npy', articulation_dict)
            '''
                articulations must be saved in a h5 file and be independent of the SimulationData module
            '''

    # for vowel in english_vowels_keys:
    #     distance_to_vowel[vowel] = np.delete(distance_to_vowel[vowel], 0, axis=0)  # Delete the infinity element

    # simulation_data.save_data(file_name + '.h5')

    np.save(file_name + '_dist_to_vowel.npy', distance_to_vowel)
    np.save(file_name + '_articulations.npy', articulation_dict)

if __name__ == '__main__' and report:

    # Experiment report and audios
    file_name_articulations = 'exploration_VDSG_articulations.npy'
    file_name_distances = 'exploration_VDSG_dist_to_vowel.npy'

    from exploration.systems.DivaStatic import DivaStatic

    english_vowels_keys = english_vowels.keys()

    system = DivaStatic()

    distance_to_unit = np.load(file_name_distances).all()
    articulations = np.load(file_name_articulations).all()

    for vowel in english_vowels_keys:
        motor_command_tmp = system.motor_command
        motor_command_tmp = articulations[vowel].motor_data.data.iloc[-1].as_matrix()
        system.setMotorCommand(motor_command_tmp)
        system.getSoundWave(play=0, save=1, file_name='vt_' + vowel + '_')
        print("Minimum distance to {} is {}".format(vowel, distance_to_unit[vowel][-1]))

    system.releaseAudioDevice()

if __name__ == '__main__' and file_conv:
    import sys
    sys.path.append("../../")  # To guarantee that the  data path is available to read the  npy file
    # Experiment report and audios
    # file_name_h5 = 'exploration_VDSG.h5'
    file_ = raw_input("\n Please, write the path+name of the file. ")  #'exploration_VDSG_articulations.npy'
    file_name_articulations = open(file_, 'r')


    file_name = 'exploration_VDSG_articulation_'

    from exploration.systems.DivaStatic import DivaStatic

    english_vowels_keys = english_vowels.keys()

    system = DivaStatic()

    articulations = np.load(file_name_articulations).all()

    for vowel in english_vowels_keys:
        articulations[vowel].save_data(file_name + vowel + '_' + '.h5')


if __name__ == '__main__' and data_set_vowels:
    ''' To generate a dataset which will be used with Diva2016a'''
    import sys
    sys.path.append("../../")

    from exploration.systems.Diva2016a import Diva2016a as System
    from exploration.data.data import SimulationData


    file_ = raw_input("\n Please, write the path+name of the file. ")  #'exploration_VDSG_articulations.npy'
    file_name_articulations = open(file_, 'r')

    file_name = 'exploration_VDSG_articulation_'

    english_vowels_keys = english_vowels.keys()

    system = System()
    vowels_data_set = SimulationData(system)

    articulations = np.load(file_name_articulations).all()

    order_of_vowels = []

    for vowel1 in english_vowels_keys:
        m1 = articulations[vowel1].motor_data.data.iloc[-1]
        m2 = np.zeros(m1.shape)

        m = np.concatenate((m1, m2)).flatten()
        system.set_action(m)
        system.executeMotorCommand()
        vowels_data_set.append_data(system)
        order_of_vowels += [[vowel1 + '_']]

    for vowel1 in english_vowels_keys:
        m1 = articulations[vowel1].motor_data.data.iloc[-1]
        for vowel2 in english_vowels_keys:
            m2 = articulations[vowel2].motor_data.data.iloc[-1]
            m = np.concatenate((m1, m2)).flatten()
            system.set_action(m)
            system.executeMotorCommand()
            vowels_data_set.append_data(system)
            order_of_vowels += [[vowel1 + '_' + vowel2]]

    vowels_data_set.save_data("vowels_dataset_1.h5")

    with open('vowels_dataset_1.txt', 'w') as file:
        for item in order_of_vowels:
            print>> file, item[0]
