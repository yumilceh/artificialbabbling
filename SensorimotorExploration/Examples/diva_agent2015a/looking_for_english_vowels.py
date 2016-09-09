'''
Created on Sept 8, 2016

@author: Juan Manuel Acevedo Valle
'''

if __name__ == '__main__':
    
    import os,sys
    sys.path.append(os.getcwd())
    from SensorimotorSystems.Diva_Proprio2016a import Diva_Proprio2016a
    from Algorithm.RndSensorimotorFunctions import get_random_motor_set
    
    english_vowels = {'i':[296.0, 2241.0, 1.0], 'I': [396.0, 1839.0, 1.0], 'e': [532.0, 1656.0, 1.0], 'ae': [667.0, 1565.0, 1.0],
    'A': [661.0, 1296.0, 1.0], 'a': [680.0, 1193.0, 1.0], 'b': [643.0, 1019.0, 1.0], 'c': [480.0, 857.0, 1.0],
    'U': [395.0, 1408.0, 1.0], 'u': [386.0, 1587.0, 1.0], 'E':  [519.0, 1408.0, 1.0]}

    english_vowels_keys = english_vowels.keys()
    
    distance_to_unit = {key: None for key in english_vowels_keys}
    articulation = {key: None for key in english_vowels_keys}
    
    
    
    n_experiments = 2 
    system = Diva_Proprio2016a()
    motor_commands = get_random_motor_set(system, n_experiments)
    
    for i in range(n_experiments):
        system.setMotorCommand(motor_commands[i,:])    
        system.getMotorDynamics()
        system.vocalize()
        #=======================================================================
        # system.getSoundWave(play=1)
        #=======================================================================
        system.getVocalizationVideo()

  
    
  