'''
Created on Sept 8, 2016

@author: Juan Manuel Acevedo Valle
'''
import os,sys
import numpy as np
import csv

#===============================================================================
# 
# if __name__ == '__main__':
#     
#     sys.path.append(os.getcwd())
#     from SensorimotorSystems.Diva_Proprio2016a import Diva_Proprio2016a
#     from Algorithm.RndSensorimotorFunctions import get_random_motor_set
#     from DataManager.SimulationData import SimulationData
#     from Algorithm.StorageDataFunctions import loadSimulationData_h5
# 
#     
#     data_file = 'simulation_data'
#     diva_output_scale=[100.0,500.0,1500.0,3000.0];
# 
#     english_vowels = {'i':[296.0, 2241.0, 1.0], 'I': [396.0, 1839.0, 1.0], 'e': [532.0, 1656.0, 1.0], 'ae': [667.0, 1565.0, 1.0],
#     'A': [661.0, 1296.0, 1.0], 'a': [680.0, 1193.0, 1.0], 'b': [643.0, 1019.0, 1.0], 'c': [480.0, 857.0, 1.0],
#     'U': [395.0, 1408.0, 1.0], 'u': [386.0, 1587.0, 1.0], 'E':  [519.0, 1408.0, 1.0]}
# 
#     english_vowels_keys = english_vowels.keys()
#     
#     distance_to_unit = {key: None for key in english_vowels_keys}
#     articulation_index = {key: None for key in english_vowels_keys}
#     for vowel in english_vowels_keys:
#         distance_to_unit[vowel] =   np.finfo(np.float64).max
#         
# 
#     
#     file_name = "_looking_vowels_1"
#     n_experiments = 50
#     
#     system = Diva_Proprio2016a()
#     
#     simulation_results = loadSimulationData_h5('simulation_data.h5', system)
#     
#     for i in range(n_experiments):
#         auditory_output = simulation_results.sensor_data.data.iloc[i]
#         
#         #=======================================================================
#         # system.plotArticulatoryEvolution([1,2,3,4,5,6,7,8,9,10,11,12,13])
#         #=======================================================================
#         if auditory_output[2]>0.1 or auditory_output[5]>0.1:
#             for vowel in english_vowels_keys:
#                 ref_formants = english_vowels[vowel]
# 
#                 #Perception window 1
#                 F1 = auditory_output[0]*diva_output_scale[1]
#                 F2 = auditory_output[1]*diva_output_scale[2]
#                 distance = np.sqrt( (F1-ref_formants[0])**2 + (F2-ref_formants[1])**2)
#                 if distance < distance_to_unit[vowel]:
#                     distance_to_unit[vowel] = distance
#                     articulation_index[vowel] = i
#                       
#                 #Perception window 2
#                 F1 = auditory_output[3]*diva_output_scale[1]
#                 F2 = auditory_output[4]*diva_output_scale[2]
#                 distance = np.sqrt( (F1-ref_formants[0])**2 + (F2-ref_formants[1])**2)
#                 if distance < distance_to_unit[vowel]:
#                     distance_to_unit[vowel] = distance
#                     articulation_index[vowel] = i
#                     
#         print('Looking for vowels. Experiment: {} of {}'.format(i+1,n_experiments))
#                 
#                 
#     
#     np.save('distances_result.npy', distance_to_unit)
#     np.save('index_result.npy', articulation_index)
#             
#     w = csv.writer(open("Distances_to_vowels.csv", "w"))
#     for key, val in distance_to_unit.items():
#         w.writerow([key, val])
#     v = csv.writer(open("Articulator_Indixes.csv", "w"))       
#     for key, val in articulation_index.items():
#         v.writerow([key, val])    
#     
#     
#     for key, val in distance_to_unit.items():
#         print("minimum distance to {} is {} for artiulation {}".format(key, val, articulation_index[key]))
#===============================================================================


#========================================================================
# ORIGINAL EXPERIMENT
#========================================================================
if __name__ == '__main__':
     
    sys.path.append(os.getcwd())
    from SensorimotorSystems.Diva_Proprio2016a import Diva_Proprio2016a
    from Algorithm.RndSensorimotorFunctions import get_random_motor_set
    from DataManager.SimulationData import SimulationData
     
 
    diva_output_scale=[100.0,500.0,1500.0,3000.0];
 
    english_vowels = {'i':[296.0, 2241.0, 1.0], 'I': [396.0, 1839.0, 1.0], 'e': [532.0, 1656.0, 1.0], 'ae': [667.0, 1565.0, 1.0],
    'A': [661.0, 1296.0, 1.0], 'a': [680.0, 1193.0, 1.0], 'b': [643.0, 1019.0, 1.0], 'c': [480.0, 857.0, 1.0],
    'U': [395.0, 1408.0, 1.0], 'u': [386.0, 1587.0, 1.0], 'E':  [519.0, 1408.0, 1.0]}
 
    english_vowels_keys = english_vowels.keys()
     
    distance_to_unit = {key: None for key in english_vowels_keys}
    articulation_dict = {key: None for key in english_vowels_keys}

 
     
    file_name = "_looking_vowels_1"
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
                distance = np.sqrt( (F1-ref_formants[0])**2 + (F2-ref_formants[1])**2)
                if distance < distance_to_unit[vowel]:
                    distance_to_unit[vowel] = distance
                    articulation_dict[vowel].appendData(system)
                       
                #Perception window 2
                F1 = auditory_output[3]*diva_output_scale[1]
                F2 = auditory_output[4]*diva_output_scale[2]
                distance = np.sqrt( (F1-ref_formants[0])**2 + (F2-ref_formants[1])**2)
                if distance < distance_to_unit[vowel]:
                    distance_to_unit[vowel] = distance
                    articulation_dict[vowel].appendData(system)
                     
        print('Looking for vowels. Experiment: {} of {}'.format(i+1,n_experiments))
                 
                 
    simulation_data.saveData('test.h5')    
     
    np.save('distances_result.npy', distance_to_unit)
    np.save('index_result.npy', articulation_dict)
                  
    for key, val in distance_to_unit.items():
        print("minimum distance to {} is {} for articulation:".format(key, val))
        print(articulation_dict[key].motor_data.data)
