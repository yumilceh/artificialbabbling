'''
Created on Feb 5, 2016
This cass defines the DIVA agent used for the CCIA 2015's paper
@author: Juan Manuel Acevedo Valle
'''
import os
import math
import numpy as np
import pymatlab as ml
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import linspace

class Diva_Proprio2015a:
    
    def __init__(self):
        motorDimensions=26
        sensorDimensions=7
        outputScale=[100.0,500.0,1500.0,3000.0];
        ts=0.01;
        self.ts=ts;
        self.time=linspace(0, .8, int(.8/ts)+1)
        self.motorDimensions=motorDimensions
        self.sensorDimensions=sensorDimensions
        self.motorCommand=[0.0] * motorDimensions
        self.sensorOutput=[0.0] * sensorDimensions
        self.matlabSesion=ml.session_factory()        
        self.matlabSesion.run('cd /home/yumilceh/eclipse_ws/Early_Development/Sensorimotor_Exploration/Sensorimotor_Systems/DIVA/') #Path to DIVA functions
        self.matlabSesion.putvalue('outputScale', outputScale)
        
    def setMotorCommand(self,motorCommand):
        self.motorCommand=motorCommand    
        
    def getMotorDynamics(self):
        ts=self.ts
        durationM1=0.25
        durationM2=0.55
        nSamples=int(0.8/ts+1)
        nSamples1=int(durationM1/ts)+1
        nSamples2=int(durationM2/ts)+1
        y_neutral=[0.0]*13
        y_neutral[11]=-0.25
        y_neutral[12]=-0.25
        y0=[0.0]*26
        y0[:13]=y_neutral
        m1=self.motorCommand[:13]
        t1=linspace(0.0,durationM1,nSamples1)
        artStates1=odeint(motorDynamics,y0,t1,args=(self,m1))
        t2=linspace(0.0,durationM2,nSamples2)
        m2=self.motorCommand[13:]
        artStates2=odeint(motorDynamics,artStates1[-1,:],t2,args=(self,m2))
        self.artStates= np.zeros((nSamples, 26))
        self.artStates[:nSamples1,:]=artStates1
        self.artStates[nSamples1-1:,:]=artStates2
    
    def vocalize(self):
        ts=self.ts;
        perceptionWindowDuration=0.15;
        perceptionTime=linspace(0,perceptionWindowDuration, int(perceptionWindowDuration/ts)+1)
        nPerceptionSamples=(len(perceptionTime))
        self.auditoryResult=[0.0]*6
        proprioceptiveAv=[0.0]*2
        self.matlabSesion.putvalue('artStates',self.artStates)
        #self.matlabSesion.run('save test1.mat artStates')
        
        #First perception time window
        for index in range(nPerceptionSamples):
            print(index)
            self.matlabSesion.run('[auditoryState, ~, ~, af]=diva_synth(transpose(artStates('+str(index+26)+',1:13)));')
            self.matlabSesion.run('auditoryState=(auditoryState./(outputScale)-1)');
            auditoryState=self.matlabSesion.getvalue('auditoryState');
            areaFunction=self.matlabSesion.getvalue('af')
            if (self.artStates[index+25,11]>0) and (self.artStates[index+25,12]>0) and (np.min(areaFunction)>0):
                self.auditoryResult[0]=self.auditoryResult[0]+(auditoryState[1]/nPerceptionSamples)
                self.auditoryResult[1]=self.auditoryResult[1]+(auditoryState[2]/nPerceptionSamples)
                self.auditoryResult[2]=self.auditoryResult[2]+(1.0/nPerceptionSamples)
            proprioceptiveAv[0]=proprioceptiveAv[0]+(np.min(areaFunction)/nPerceptionSamples)
            
        #Second perception time window
        for index in range(nPerceptionSamples):             
            print(index)
            self.matlabSesion.run('[auditoryState, ~, ~, af]=diva_synth(transpose(artStates('+str(index+66)+',1:13)));')
            self.matlabSesion.run('auditoryState=(auditoryState./(outputScale)-1)');
            auditoryState=self.matlabSesion.getvalue('auditoryState');
            areaFunction=self.matlabSesion.getvalue('af')
            if (self.artStates[index+65,11]>0) and (self.artStates[index+65,12]>0) and (np.min(areaFunction)>0):
                self.auditoryResult[3]=self.auditoryResult[3]+(auditoryState[1]/nPerceptionSamples)
                self.auditoryResult[4]=self.auditoryResult[4]+(auditoryState[2]/nPerceptionSamples)
                self.auditoryResult[5]=self.auditoryResult[5]+(1.0/nPerceptionSamples)
            proprioceptiveAv[1]=proprioceptiveAv[1]+(np.min(areaFunction)/nPerceptionSamples)
                
        self.proprioceptiveResult=0
        if((proprioceptiveAv[0]<0.0) or (proprioceptiveAv[1]<0.0)):
            self.proprioceptiveResult=1
              
            
    def plotArticulatoryEvolution(self,arts):
        for index in range(len(arts)):
            print(arts[index]-1)
            plt.plot(self.time,self.artStates[:,arts[index]-1])
            plt.hold(True)
        plt.show()
        
    def plotAuditoryOutput(self):
        juan=1
        
def motorDynamics(y,t,self,m):
    dumpingFactor=1.01
    w0=2*math.pi/0.8
    
    dy1=y[13]
    dy2=y[14]
    dy3=y[15]
    dy4=y[16]
    dy5=y[17]
    dy6=y[18]
    dy7=y[19]
    dy8=y[20]
    dy9=y[21]
    dy10=y[22]
    dy11=y[23]
    dy12=y[24]
    dy13=y[25]
        
    dy14=-2*dumpingFactor*w0*y[13]-(pow(w0,2))*y[0]+(pow(w0,2))*m[0]
    dy15=-2*dumpingFactor*w0*y[14]-(pow(w0,2))*y[1]+(pow(w0,2))*m[1]
    dy16=-2*dumpingFactor*w0*y[15]-(pow(w0,2))*y[2]+(pow(w0,2))*m[2]
    dy17=-2*dumpingFactor*w0*y[16]-(pow(w0,2))*y[3]+(pow(w0,2))*m[3]
    dy18=-2*dumpingFactor*w0*y[17]-(pow(w0,2))*y[4]+(pow(w0,2))*m[4]
    dy19=-2*dumpingFactor*w0*y[18]-(pow(w0,2))*y[5]+(pow(w0,2))*m[5]
    dy20=-2*dumpingFactor*w0*y[19]-(pow(w0,2))*y[6]+(pow(w0,2))*m[6]
    dy21=-2*dumpingFactor*w0*y[20]-(pow(w0,2))*y[7]+(pow(w0,2))*m[7]
    dy22=-2*dumpingFactor*w0*y[21]-(pow(w0,2))*y[8]+(pow(w0,2))*m[8]
    dy23=-2*dumpingFactor*w0*y[22]-(pow(w0,2))*y[9]+(pow(w0,2))*m[9]
    dy24=-2*dumpingFactor*w0*y[23]-(pow(w0,2))*y[10]+(pow(w0,2))*m[10]
    dy25=-2*dumpingFactor*w0*y[24]-(pow(w0,2))*y[11]+(pow(w0,2))*m[11]
    dy26=-2*dumpingFactor*w0*y[25]-(pow(w0,2))*y[12]+(pow(w0,2))*m[12]
    
    return [dy1,dy2,dy3,dy4,dy5,dy6,dy7,dy8,dy9,dy10,dy11,dy12,dy13,dy14,dy15,dy16,dy17,dy18,dy19,dy20,dy21,dy22,dy23,dy24,dy25,dy26]
    
