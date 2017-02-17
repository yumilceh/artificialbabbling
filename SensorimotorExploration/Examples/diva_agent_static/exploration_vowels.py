'''
Created on Feb 16, 2017

@author: Juan Manuel Acevedo Valle
'''
import sys, datetime
import numpy as np

'''
#=======================================================================================
#
# ORIGINAL EXPERIMENT: Script to look for motor configurations producing english 
# vowels using a random approach. 
#
#=======================================================================================
'''
explore = True
report = False
now = datetime.datetime.now().strftime("DSG_%Y_%m_%d_%H_%M_")

file_name = "exploration_vowels_" + now  

if __name__ == '__main__' and explore:

    sys.path.append("../../")
    from SensorimotorSystems.DivaStatic import DivaStatic
    from Algorithm.utils.RndSensorimotorFunctions import get_random_motor_set
    from DataManager.SimulationData import SimulationData


    diva_output_scale=[100.0,500.0,1500.0,3000.0]
    '''If normalization is needed this are the DIVA scales,
    important to mention that for DivaStatic, format frequencies
    are read unnormilized'''  

    english_vowels = {'i':[296.0, 2241.0, 1.0], 'I': [396.0, 1839.0, 1.0], 'e': [532.0, 1656.0, 1.0], 'ae': [667.0, 1565.0, 1.0],
    'A': [661.0, 1296.0, 1.0], 'a': [680.0, 1193.0, 1.0], 'b': [643.0, 1019.0, 1.0], 'c': [480.0, 857.0, 1.0],
    'U': [395.0, 1408.0, 1.0], 'u': [386.0, 1587.0, 1.0], 'E':  [519.0, 1408.0, 1.0]}

    english_vowels_keys = english_vowels.keys()

    distance_to_vowel = {key: None for key in english_vowels_keys}
    articulation_dict = {key: None for key in english_vowels_keys}


    n_experiments = 1000000
    n_save = 50000 #Save data each n_save samples

    system = DivaStatic()

    for vowel in english_vowels_keys:
        distance_to_vowel[vowel] = [np.infty] 
        articulation_dict[vowel] = SimulationData(system)

    simulation_data = SimulationData(system)

    min_motor_values=np.array([-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,0.2,0.2,0.2])
    max_motor_values=np.array([2,2,2,2,2,2,2,2,2,2,1,1,1])

    motor_commands = get_random_motor_set(system, n_experiments, min_values=min_motor_values, max_values=max_motor_values)

    for i in range(n_experiments):
        system.setMotorCommand(motor_commands[i,:])
        system.executeMotorCommand()

        simulation_data.appendData(system)
        auditory_output = system.sensor_out

        if auditory_output[1]>0.00001:
            for vowel in english_vowels_keys:
                ref_formants = english_vowels[vowel]

                F1 = auditory_output[1]
                F2 = auditory_output[2]
                distance = np.sqrt( (F1-ref_formants[0])**2 + (F2-ref_formants[1])**2)
                if distance < distance_to_vowel[vowel][-1]:
                    distance_to_vowel[vowel] = np.append(distance_to_vowel[vowel],distance)
                    articulation_dict[vowel].appendData(system)

        print('Looking for vowels. Experiment: {} of {}'.format(i+1,n_experiments))
        
        if (np.mod(i,n_save) == 0): 
            simulation_data.saveData(file_name +  '.h5')
        
            np.save(file_name +  '_dist_to_vowel.npy', distance_to_vowel)
            np.save(file_name +  '_articulations.npy', articulation_dict)
            
            
    #distance_to_unit = np.delete(distance, 0, axis=0) #Delete the infty element
    
    simulation_data.saveData(file_name +  '.h5')

    np.save(file_name +  '_dist_to_vowel.npy', distance_to_vowel)
    np.save(file_name +  '_articulations.npy', articulation_dict)
                  
    #--------------------------------- for key, val in distance_to_unit.items():
        #---------------- print("minimum distance to {} is {}".format(key, val))
        #------------------------- print(articulation_dict[key].motor_data.data)
        
#=======================================================================================
#
# Experiment report and videos
#
#=======================================================================================
if __name__ == '__main__' and report:
     
    sys.path.append("../../")
    from SensorimotorSystems.Diva_Proprio2016a import Diva_Proprio2016a

    #===========================================================================
    # data_file = 'looking_for_english_vowel_results/Experiment_1/looking_english_vowels.h5'
    #===========================================================================
 
    english_vowels = {'i':[296.0, 2241.0, 1.0], 'I': [396.0, 1839.0, 1.0], 'e': [532.0, 1656.0, 1.0], 'ae': [667.0, 1565.0, 1.0],
    'A': [661.0, 1296.0, 1.0], 'a': [680.0, 1193.0, 1.0], 'b': [643.0, 1019.0, 1.0], 'c': [480.0, 857.0, 1.0],
    'U': [395.0, 1408.0, 1.0], 'u': [386.0, 1587.0, 1.0], 'E':  [519.0, 1408.0, 1.0]}
 
    english_vowels_keys = english_vowels.keys()
              
    system = Diva_Proprio2016a()    
    
    distance_to_unit = np.load('looking_for_english_vowel_results/Experiment_1/distances_result_lev.npy').all()
    articulations = np.load('looking_for_english_vowel_results/Experiment_1/articulations_data_lev.npy').all()
    best_perception_window = np.load('looking_for_english_vowel_results/Experiment_1/best_perception_window_lev.npy').all()
        
    for vowel in english_vowels_keys:
        motor_command_tmp = system.motor_command
        if best_perception_window[vowel]==1:
            motor_command_tmp[0:13] = articulations[vowel].motor_data.data.iloc[-1][0:13].as_matrix()
            motor_command_tmp[13:] = articulations[vowel].motor_data.data.iloc[-1][0:13].as_matrix()
            system.setMotorCommand(motor_command_tmp)
        if best_perception_window[vowel]==2:
            motor_command_tmp[0:13] = articulations[vowel].motor_data.data.iloc[-1][13:].as_matrix()
            motor_command_tmp[13:] = articulations[vowel].motor_data.data.iloc[-1][13:].as_matrix()
            system.setMotorCommand(motor_command_tmp)
        system.executeMotorCommand()
        system.getVocalizationVideo(show=0, file_name='looking_for_english_vowel_results/Experiment_1/vt'+vowel) #no extension in files        

    for key, val in distance_to_unit.items():
        print("minimum distance to {} is {}".format(key, val))
           
