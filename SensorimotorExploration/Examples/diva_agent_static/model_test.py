'''
Created on Feb 15, 2017

@author: Juan Manuel Acevedo Valle
'''

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    from numpy import array as arr
    sys.path.append('../../')
    from SensorimotorSystems.DivaStatic import DivaStatic 

    diva_system=DivaStatic()
    
    #Test motor execution
    diva_system.setMotorCommand(arr([3, 0, 2, 0.1, 0, 0, 0, 0, 0, 0, 1.0, 1., 1.]))    
    diva_system.executeMotorCommand()
    print(diva_system.sensor_out)
    print(diva_system.somato_out)
    
    #Test plot vocal tract shape
    fig, ax = diva_system.plotVocalTractShape()
    plt.show()
    
    #Test generating sound
    diva_system.getSoundWave(play=1,save=0)
    x = input()
    diva_system.releaseAudioDevice()
    