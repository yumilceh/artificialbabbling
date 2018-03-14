'''
Created on Sept 8, 2016

@author: Juan Manuel Acevedo Valle
'''
import os,sys
import numpy as np
import time
#=======================================================================================
#
# Experimental analysis
#
#=======================================================================================
if __name__ == '__main__' and True:
    from SensorimotorSystems.Diva_Proprio2016a import Diva_Proprio2015a
    import h5py
    
    diva_output_scale=[100.0,500.0,1500.0,3000.0]
    #===========================================================================
    # directories = ['data_test/']
    #===========================================================================
    #===========================================================================
    # '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_no_Proprio_0/',
    #                '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_Proprio_0/',
    #===========================================================================
    directories = ['/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_no_Proprio_1/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_Proprio_1/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_no_Proprio_2/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_Proprio_2/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_no_Proprio_3/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_Proprio_3/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_no_Proprio_4/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_Proprio_4/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_no_Proprio_6/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_Proprio_6/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_no_Proprio_7/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_Proprio_7/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_no_Proprio_8/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_Proprio_8/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_no_Proprio_9/',
                   '/media/yumilceh/yumilcehBackup/ExperimentsIEEETCDS2016/EVD_Proprio_9/']
                     

    for directory in directories:    
        print('Working in ' + directory)
        mat = h5py.File(directory + 'SMdata.mat','r')
        data = np.array(mat.get('SMdata'))
        n_samples = data.shape[1]
        inverse_indices = range(n_samples-1, 1000, -500)
        motor_data = data[6:,inverse_indices]
        sensor_data = data[0:6,inverse_indices]
        
        system = Diva_Proprio2015a()    
    
        #--------------------------------------------- for i in inverse_indices:
        for i in range(motor_data.shape[1]):
            motor_command_tmp = system.motor_command
            
            motor_command_tmp[0:7] = motor_data[0:7,i].flatten()
            motor_command_tmp[10:12] = motor_data[7:9,i].flatten()
            motor_command_tmp[13:20] = motor_data[9:16,i].flatten()
            motor_command_tmp[23:-1] = motor_data[16:,i].flatten()
    
            system.set_action(motor_command_tmp)
            system.execute_action()
            
            try:
                system.get_sound(play=1)
            except:
                print('Something wnet wrong with this configuration. No sound played.')
            
            time.sleep(0.5)
            print('Vocalization (inverse_order): {}'.format(i))
