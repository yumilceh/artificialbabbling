'''
Created on Feb 5, 2016

@author: yumilceh
'''
from Sensorimotor_Systems.Diva_Proprio2015a import Diva_Proprio2015a

if __name__ == '__main__':
    divaAgent=Diva_Proprio2015a()
    divaAgent.setMotorCommand([2,0,2,0,0,0,0,0,0,0,0,0.7,0.7,-3,0,2,0,0,0,0,0,0,0,0,0.7,0.7])
    divaAgent.getMotorDynamics()
    divaAgent.vocalize()
    print(divaAgent.auditoryResult)
    print(divaAgent.proprioceptiveResult)
    #divaAgent.plotArticulatoryEvolution([1,3,12,13])
    #print(divaAgent.artStates)
    #print(divaAgent.artStates[:,10])
    #print(len(divaAgent.time))