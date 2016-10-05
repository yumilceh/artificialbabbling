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
    from SensorimotorSystems.Diva_Proprio2016a import Diva_Proprio2015a
    import h5py
    
    diva_output_scale=[100.0,500.0,1500.0,3000.0]

    #--------------- directories = ['ExperimentsIEEETCDS2016/EVD_no_Proprio_0/',
                   #--------------- 'ExperimentsIEEETCDS2016/EVD_no_Proprio_1/',
                   #--------------- 'ExperimentsIEEETCDS2016/EVD_no_Proprio_2/',
                   #--------------- 'ExperimentsIEEETCDS2016/EVD_no_Proprio_3/',
                   #--------------- 'ExperimentsIEEETCDS2016/EVD_no_Proprio_4/',
                   #--------------- 'ExperimentsIEEETCDS2016/EVD_no_Proprio_6/',
                   #--------------- 'ExperimentsIEEETCDS2016/EVD_no_Proprio_7/',
                   #--------------- 'ExperimentsIEEETCDS2016/EVD_no_Proprio_8/',
                   #--------------- 'ExperimentsIEEETCDS2016/EVD_no_Proprio_9/',
                   #------------------ 'ExperimentsIEEETCDS2016/EVD_Proprio_0/',
                   #------------------ 'ExperimentsIEEETCDS2016/EVD_Proprio_1/',
                   #------------------ 'ExperimentsIEEETCDS2016/EVD_Proprio_2/',
                   #------------------ 'ExperimentsIEEETCDS2016/EVD_Proprio_3/',
                   #------------------ 'ExperimentsIEEETCDS2016/EVD_Proprio_4/',
                   #------------------ 'ExperimentsIEEETCDS2016/EVD_Proprio_6/',
                   #------------------ 'ExperimentsIEEETCDS2016/EVD_Proprio_7/',
                   #------------------ 'ExperimentsIEEETCDS2016/EVD_Proprio_8/',
                   #------------------ 'ExperimentsIEEETCDS2016/EVD_Proprio_9/',
                   # 'ExperimentsIEEETCDS2016/Special_EVD_Proprio_5/EVD_no_Proprio_5',
                   # 'ExperimentsIEEETCDS2016/Special_EVD_Proprio_5/EVD_Proprio_5']
                   
    directories = ['ExperimentsIEEETCDS2016/EVD_Proprio_9/',
                   'ExperimentsIEEETCDS2016/Special_EVD_Proprio_5/EVD_no_Proprio_5',
                   'ExperimentsIEEETCDS2016/Special_EVD_Proprio_5/EVD_Proprio_5']
    
    for directory in directories:    
        mat = h5py.File(directory + 'SMdata.mat','r')
        data = np.array(mat.get('SMdata'))
        motor_data = data[6:,-50001:]
        sensor_data = data[0:6,-50001:]
        
        english_vowels = {'i':[296.0, 2241.0, 1.0], 'I': [396.0, 1839.0, 1.0], 'e': [532.0, 1656.0, 1.0], 'ae': [667.0, 1565.0, 1.0],
        'A': [661.0, 1296.0, 1.0], 'a': [680.0, 1193.0, 1.0], 'b': [643.0, 1019.0, 1.0], 'c': [480.0, 857.0, 1.0],
        'U': [395.0, 1408.0, 1.0], 'u': [386.0, 1587.0, 1.0], 'E':  [519.0, 1408.0, 1.0]}
     
        english_vowels_keys = english_vowels.keys()
        
        distance_to_unit = {key: None for key in english_vowels_keys}
        articulation_dict = {key: None for key in english_vowels_keys}
        best_perception_window = {key: None for key in english_vowels_keys}
                  
        system = Diva_Proprio2015a()    
            
        n_experiments = motor_data.shape[1]
        
        distance_to_unit = {key: None for key in english_vowels_keys}
        articulation_index_dict = {key: None for key in english_vowels_keys}
        best_perception_window = {key: None for key in english_vowels_keys}
    
    
        system = Diva_Proprio2015a()
    
        for vowel in english_vowels_keys:
            distance_to_unit[vowel] =   [np.infty] 
            best_perception_window[vowel] = []
            articulation_index_dict[vowel] = [np.infty]
    
        for i in range(n_experiments):
        
            auditory_output = sensor_data[:,i].flatten()
            
            if auditory_output[2]>0.1 or auditory_output[5]>0.1:
                for vowel in english_vowels_keys:
                    ref_formants = english_vowels[vowel]
            
                    #Perception window 1
                    F1 = auditory_output[0]*diva_output_scale[1]
                    F2 = auditory_output[1]*diva_output_scale[2]
                    distance1 = np.sqrt( (F1-ref_formants[0])**2 + (F2-ref_formants[1])**2)
                    if distance1 < distance_to_unit[vowel][-1]:
                        distance_to_unit[vowel] = np.append(distance_to_unit[vowel],distance1)
                        articulation_index_dict[vowel] = np.append(articulation_index_dict[vowel], i)
            
                    #Perception window 2
                    F1 = auditory_output[3]*diva_output_scale[1]
                    F2 = auditory_output[4]*diva_output_scale[2]
                    distance2 = np.sqrt( (F1-ref_formants[0])**2 + (F2-ref_formants[1])**2)
                    if distance2 < distance_to_unit[vowel][-1]:
                        distance_to_unit[vowel] = np.append(distance_to_unit[vowel],distance2)
                        articulation_index_dict[vowel] = np.append(articulation_index_dict[vowel], i)
            
            
                    if distance1 < distance2:
                        best_perception_window[vowel] =  np.append(best_perception_window[vowel],1)
                    else:
                        best_perception_window[vowel] = np.append(best_perception_window[vowel],2)  
            
            print('Looking for vowels. Experiment: {} of {}'.format(i+1,n_experiments))
        
        for vowel in english_vowels_keys:
            distance_to_unit[vowel] =   np.delete(distance_to_unit[vowel], 0, axis=0) #Delete the infty element
            articulation_index_dict[vowel] = np.delete( articulation_index_dict[vowel], 0, axis=0) #Delete the infty element
          
                
        np.save(directory + 'distances_result_lev.npy', distance_to_unit)
        np.save(directory + 'articulations_data_lev.npy', articulation_dict)
        np.save(directory + 'best_perception_window_lev.npy', best_perception_window)
            
        for key, val in distance_to_unit.items():
            print("minimum distance to {} is {}".format(key, val))
    
    
        for vowel in english_vowels_keys:
            index = articulation_index_dict[vowel][-1]
            motor_command_tmp = system.motor_command
            
            motor_command_tmp[0:7] = motor_data[0:7,index].flatten()
            motor_command_tmp[10:12] = motor_data[7:9,index].flatten()
            motor_command_tmp[13:20] = motor_data[9:16,index].flatten()
            motor_command_tmp[23:-1] = motor_data[16:,index].flatten()
    
            system.setMotorCommand(motor_command_tmp)
            system.executeMotorCommand()
            system.getVocalizationVideo(show=0, file_name=directory + 'vt'+vowel) #no extension in files
    
       