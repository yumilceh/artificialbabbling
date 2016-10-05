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

    #===========================================================================
    # directories = ['ExperimentsIEEETCDS2016/EVD_no_Proprio_0/',
    #                'ExperimentsIEEETCDS2016/EVD_no_Proprio_1/',
    #                'ExperimentsIEEETCDS2016/EVD_no_Proprio_2/',
    #                'ExperimentsIEEETCDS2016/EVD_no_Proprio_3/',
    #                'ExperimentsIEEETCDS2016/EVD_no_Proprio_4/',
    #                'ExperimentsIEEETCDS2016/EVD_no_Proprio_6/',
    #                'ExperimentsIEEETCDS2016/EVD_no_Proprio_7/',
    #                'ExperimentsIEEETCDS2016/EVD_no_Proprio_8/',
    #                'ExperimentsIEEETCDS2016/EVD_no_Proprio_9/',
    #                'ExperimentsIEEETCDS2016/EVD_Proprio_0/',
    #                'ExperimentsIEEETCDS2016/EVD_Proprio_1/',
    #                'ExperimentsIEEETCDS2016/EVD_Proprio_2/',
    #                'ExperimentsIEEETCDS2016/EVD_Proprio_3/',
    #                'ExperimentsIEEETCDS2016/EVD_Proprio_4/',
    #                'ExperimentsIEEETCDS2016/EVD_Proprio_6/',
    #                'ExperimentsIEEETCDS2016/EVD_Proprio_7/',
    #                'ExperimentsIEEETCDS2016/EVD_Proprio_8/',
    #                'ExperimentsIEEETCDS2016/EVD_Proprio_9/',
    #                'ExperimentsIEEETCDS2016/Special_EVD_Proprio_5/EVD_no_Proprio_5/',
    #                'ExperimentsIEEETCDS2016/Special_EVD_Proprio_5/EVD_Proprio_5/']
    #===========================================================================
                   
    directories = [ 'ExperimentsIEEETCDS2016/Special_EVD_Proprio_5/EVD_no_Proprio_5/',
                    'ExperimentsIEEETCDS2016/Special_EVD_Proprio_5/EVD_Proprio_5/']

    for directory in directories:    
        mat = h5py.File(directory + 'SMdata.mat','r')
        data = np.array(mat.get('SMdata'))
        motor_data = data[6:,-50001:]
        sensor_data = data[0:6,-50001:]
        
        n_experiments = 20
                  
        system = Diva_Proprio2015a()    
    
        indeces = np.random.randint(0,50000,n_experiments)
    
        for i in range(n_experiments):
            index = indeces[i]
            motor_command_tmp = system.motor_command
            
            motor_command_tmp[0:7] = motor_data[0:7,index].flatten()
            motor_command_tmp[10:12] = motor_data[7:9,index].flatten()
            motor_command_tmp[13:20] = motor_data[9:16,index].flatten()
            motor_command_tmp[23:-1] = motor_data[16:,index].flatten()
    
            system.setMotorCommand(motor_command_tmp)
            system.executeMotorCommand()
            system.getVocalizationVideo(show=0, file_name=directory + 'vt' + str(i) ) #no extension in files
    
         print('Working in ' + directory)