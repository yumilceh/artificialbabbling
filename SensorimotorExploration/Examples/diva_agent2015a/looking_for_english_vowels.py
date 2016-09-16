'''
Created on Sept 8, 2016

@author: Juan Manuel Acevedo Valle
'''
import os,sys
import numpy as np

#=======================================================================================
#
# Experiment report and videos
#
#=======================================================================================
if __name__ == '__main__' and True:
     
    sys.path.append(os.getcwd())
    from SensorimotorSystems.Diva_Proprio2016a import Diva_Proprio2016a

    data_file = 'looking_for_english_vowel_results/looking_english_vowels.h5'
 
    english_vowels = {'i':[296.0, 2241.0, 1.0], 'I': [396.0, 1839.0, 1.0], 'e': [532.0, 1656.0, 1.0], 'ae': [667.0, 1565.0, 1.0],
    'A': [661.0, 1296.0, 1.0], 'a': [680.0, 1193.0, 1.0], 'b': [643.0, 1019.0, 1.0], 'c': [480.0, 857.0, 1.0],
    'U': [395.0, 1408.0, 1.0], 'u': [386.0, 1587.0, 1.0], 'E':  [519.0, 1408.0, 1.0]}
 
    english_vowels_keys = english_vowels.keys()
              
    system = Diva_Proprio2016a()    
    
    distance_to_unit = np.load('distances_result_lev.npy').all()
    articulations = np.load('articulations_data_lev.npy').all()
    best_perception_window = np.load('best_perception_window_lev.npy').all()
    
    
    best_perception_window = distance_to_unit
    
    for vowel in english_vowels_keys:
        ref_formants = english_vowels[vowel]
        system.setMotorCommand(articulations[vowel].motor_data.data.iloc[-1].as_matrix())
        system.executeMotorCommand()
        system.getVocalizationVideo(show=0, file_name='vt'+vowel) #no extension in files        
             
#=======================================================================================
#
# ORIGINAL EXPERIMENT
#
#=======================================================================================

if __name__ == '__main__' and False:

    sys.path.append(os.getcwd())
    from SensorimotorSystems.Diva_Proprio2016a import Diva_Proprio2016a
    from Algorithm.RndSensorimotorFunctions import get_random_motor_set
    from DataManager.SimulationData import SimulationData


    diva_output_scale=[100.0,500.0,1500.0,3000.0]

    english_vowels = {'i':[296.0, 2241.0, 1.0], 'I': [396.0, 1839.0, 1.0], 'e': [532.0, 1656.0, 1.0], 'ae': [667.0, 1565.0, 1.0],
    'A': [661.0, 1296.0, 1.0], 'a': [680.0, 1193.0, 1.0], 'b': [643.0, 1019.0, 1.0], 'c': [480.0, 857.0, 1.0],
    'U': [395.0, 1408.0, 1.0], 'u': [386.0, 1587.0, 1.0], 'E':  [519.0, 1408.0, 1.0]}

    english_vowels_keys = english_vowels.keys()

    distance_to_unit = {key: None for key in english_vowels_keys}
    articulation_dict = {key: None for key in english_vowels_keys}



    file_name = 'looking_for_english_vowel_results/looking_english_vowels.h5'
    n_experiments = 500000

    system = Diva_Proprio2016a()

    for vowel in english_vowels_keys:
        distance_to_unit[vowel] =   np.finfo(np.float64).max
        articulation_dict[vowel] = SimulationData(system)

    simulation_data = SimulationData(system)

    min_motor_values=np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0.5,0.5,0.5]*2)
    max_motor_values=np.array([1.1,1,1,1,1,1,1,1,1,1,1,1,1]*2)

    motor_commands = get_random_motor_set(system, n_experiments, min_values=min_motor_values, max_values=max_motor_values)

    for i in range(n_experiments):
        system.setMotorCommand(motor_commands[i,:])
        system.getMotorDynamics()
        system.vocalize()

        simulation_data.appendData(system)
        auditory_output = system.sensorOutput

        #=======================================================================
        # system.plotArticulatoryEvolution([1,2,3,4,5,6,7,8,9,10,11,12,13])
        #=======================================================================

        if auditory_output[2]>0.1 or auditory_output[5]>0.1:
            for vowel in english_vowels_keys:
                ref_formants = english_vowels[vowel]

                #Perception window 1
                F1 = auditory_output[0]*diva_output_scale[1]
                F2 = auditory_output[1]*diva_output_scale[2]
                distance1 = np.sqrt( (F1-ref_formants[0])**2 + (F2-ref_formants[1])**2)
                if distance1 < distance_to_unit[vowel]:
                    distance_to_unit[vowel] = distance1
                    articulation_dict[vowel].appendData(system)

                #Perception window 2
                F1 = auditory_output[3]*diva_output_scale[1]
                F2 = auditory_output[4]*diva_output_scale[2]
                distance2 = np.sqrt( (F1-ref_formants[0])**2 + (F2-ref_formants[1])**2)
                if distance2 < distance_to_unit[vowel]:
                    distance_to_unit[vowel] = distance2
                    articulation_dict[vowel].appendData(system)


                if distance1 < distance2:
                    best_perception_window[vowel] = 1
                else:
                    best_perception_window[vowel] = 2  

        print('Looking for vowels. Experiment: {} of {}'.format(i+1,n_experiments))


    simulation_data.saveData('looking_english_vowels.h5')

    np.save('distances_result_lev.npy', distance_to_unit)
    np.save('articulations_data_lev.npy', articulation_dict)
    np.save('best_perception_window_lev.npy', best_perception_window)
    
    for key, val in distance_to_unit.items():
        print("minimum distance to {} is {} for articulation:".format(key, val))
        print(articulation_dict[key].motor_data.data)
