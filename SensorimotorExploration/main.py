'''
Created on Feb 5, 2016

@author: yumilceh
'''
from Agent.SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a

import numpy as np

if __name__ == '__main__':
    divaAgent=Diva_Proprio2015a()
    divaAgent.setMotorCommand([2,0,2,0,0,0,0,0,0,0,0,0.7,0.7,-3,0,2,0,0,0,0,0,0,0,0,0.7,0.7])
    
    
    divaAgent.getMotorDynamics()
    divaAgent.vocalize()
    divaAgent.plotAuditoryOutput([1,2,3])
    divaAgent.getSoundWave(1)
    divaAgent.plotSoundWave() 
    divaAgent.playSoundWave()
    
    
    '''samples=100
    for index in range(samples):
        print(index)
        divaAgent.getMotorDynamics();
        divaAgent.vocalize();
        sample=np.concatenate((divaAgent.motorCommand[0:7],divaAgent.motorCommand[11:13],divaAgent.motorCommand[13:20],divaAgent.motorCommand[24:26],divaAgent.auditoryResult))
    '''
    
    #print(sensorimotorData.data)
    # print(divaAgent.auditoryResult)
    # print(divaAgent.proprioceptiveResult)
    # divaAgent.plotArticulatoryEvolution([1,3,12,13])
    # print(divaAgent.artStates)
    # print(divaAgent.artStates[:,10])
    # print(len(divaAgent.time))