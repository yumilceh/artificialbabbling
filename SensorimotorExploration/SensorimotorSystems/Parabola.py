'''
Created on Feb 5, 2016
This sensorimor system defines the DIVA agent used for the CCIA 2015's paper
@author: Juan Manuel Acevedo Valle
'''

#import sys
#import wave
import math
import numpy as np
from matplotlib import pyplot as plt

#from matplotlib.pyplot import autoscale
#from matplotlib.animation import Animation
#from scipy.interpolate.interpolate_wrapper import block
class CustomObject:
    def __init__(self):
        pass

class ConstrainedParabolicArea:
    
    def __init__(self):
        
        a = 2.0
        b = 3.0
        c = 3.0
        d = 3.5
        e = 1.5
        m1 = 1
        m2 = 1
        
        
        motor_names = ['M1', 'M2']
        sensor_names = ['S1', 'S2']
        somato_names = ['P1']
        n_motor = 2
        n_sensor = 2
        n_somato = 1
        
        
        min_motor_values = np.array([ 0.0, 0.0 ])
        max_motor_values = np.array([ b*2.0, b*2.0])
        
        min_sensor_values = np.array([0.0, 0.0])
        max_sensor_values = np.array([b*2.0, b**2])
        
        min_somato_values = np.array([0])
        max_somato_values = np.array([1])
        somato_threshold = np.array([0.6])

        self.params = CustomObject()
        self.params.a = a
        self.params.b = b
        self.params.c = c
        self.params.d = d
        self.params.e = e
        self.params.m1 = m1
        self.params.m2 = m2
        
        self.n_motor = n_motor
        self.n_sensor = n_sensor
        self.n_somato = n_somato
        self.motor_names = motor_names
        self.sensor_names = sensor_names
        self.somato_names = somato_names
        
        self.min_motor_values = min_motor_values
        self.max_motor_values = max_motor_values
        self.min_sensor_values = min_sensor_values
        self.max_sensor_values = max_sensor_values
        self.min_somato_values = min_somato_values
        self.max_somato_values = max_somato_values
        self.somato_threshold = somato_threshold 

        self.motor_command =np.array( [0.0] * n_motor )
        self.sensorOutput = np.array( [0.0] * n_sensor )
        self.sensor_goal = np.array([ 0.0] * n_sensor )
        self.somatoOutput = np.array( [0.0] * n_somato )
        self.competence_result = 0.0
        
    def setMotorCommand(self,motor_command):
        self.motor_command = self.boundMotorCommand(motor_command)    
    
    def executeMotorCommand_unconstrained(self):
        self.motor_command = self.boundMotorCommand(self.motor_command)    
                 
        a = self.params.a
        b = self.params.b
        c = self.params.c
        
        self.somatoOutput = 0.0
        self.sensorOutput[0] = self.motor_command[0]
        self.sensorOutput[1] = math.pow(self.motor_command[1]-b,2.0)
        
    def executeMotorCommand(self):
        self.motor_command = self.boundMotorCommand(self.motor_command)    
                 
        a = self.params.a
        b = self.params.b
        c = self.params.c
        d = self.params.d
        e = self.params.e
        
        m1 = self.params.m1
        m2 = self.params.m2
        
        r = c - a
        
        self.somatoOutput = 0.0
        self.sensorOutput[0] = self.motor_command[0]
        self.sensorOutput[1] = math.pow(self.motor_command[1]-b,2.0)
        
        x=self.sensorOutput[0]
        y=self.sensorOutput[1]
        
        point = CustomObject()
        point.x = x
        point.y = y
        
        #Checking inner Parabolic Region Condition
        parabola = CustomObject()        
        parabola.a = 1.0
        parabola.b = -2.0*b
        parabola.c = b**2
        
        if math.pow(self.motor_command[0]-b,2.0) > self.sensorOutput[1]:
            x, y = closestPointInParabola(parabola, point)
            point.x = x
            point.y = y

        ##Checking if the sensorimotor result is insede of the constrained circle
        circle = CustomObject()
        circle.x_c = b
        circle.y_c = a
        circle.r = r
                
        circle_condition = math.pow(x - b , 2.0) + math.pow(y - a , 2.0)
        if circle_condition < math.pow(r, 2.0):                       
            x, y = closestPointInCircle(circle, point)
            point.x = x
            point.y = y
 
            
        ## Checking if the sensorimotor result is inside of thecontrained region between two parallel lines
        up_line = CustomObject()
        up_line.y_0 = d 
        up_line.m = m1
        
        down_line = CustomObject()
        down_line.y_0 = e 
        down_line.m = m2
            
        if (checkLineCondition(up_line, point) == -1 and checkLineCondition(down_line, point) == 1):
            x1, y1, distance1 = closestPointToLine(up_line, point)
            x2, y2, distance2 = closestPointToLine(down_line, point)
            
            if distance1 >= distance2:
                if math.pow(x2-b,2.0) < y2:
                    x=x2
                    y=y2
                else:
                    x=x1
                    y=y1
            else:
                if math.pow(x1-b,2.0) < y1:
                    x=x1
                    y=y1
                else:
                    x=x2
                    y=y2
            point.x = x
            point.y = y
        self.sensorOutput[0] = x
        self.sensorOutput[1] = y
        
    def boundMotorCommand(self, motor_command):
        n_motor=self.n_motor
        min_motor_values = self.min_motor_values
        max_motor_values = self.max_motor_values
        
        for i in range(n_motor):
            if (motor_command[i] < min_motor_values[i]):
                motor_command[i] = min_motor_values[i]
                
            elif (motor_command[i] > max_motor_values[i]):
                motor_command[i] = max_motor_values[i]
                
        return motor_command

    def drawSystem(self, fig, axes):
        
        min_sensor_values = self.min_sensor_values
        max_sensor_values = self.max_sensor_values
        
        a = self.params.a
        b = self.params.b
        c = self.params.c
        d = self.params.d
        e = self.params.e
        
        m1 = self.params.m1
        m2 = self.params.m2
        
        r = c - a
        
        x_p = np.linspace(min_sensor_values[0], max_sensor_values[0], 100)
        y_p = (x_p - b)**2

        
        circle = plt.Circle((b,a),r,color='red')
        
        x_l1 = np.linspace(min_sensor_values[0], max_sensor_values[0], 100)
        y_l1 = d + m1*x_l1
        
        x_l2 = np.linspace(min_sensor_values[0], max_sensor_values[0], 100)
        y_l2 = e + m2*x_l2
        
        plt.figure(fig.number)
        plt.sca(axes)    
        
        
        plt.plot(x_l1, y_l1, "--r")
        plt.plot(x_l2, y_l2, "--r")        
        gap = d - e
        for i in range(100):
            plt.plot(x_l2, y_l2+i*gap/100, "r")

        axes.add_artist(circle)
        axes.set_xlim([-3.0, 9.0])
        axes.set_ylim([-1.0, 11.0])
        

        
        plt.plot(x_p, y_p, "b")
        return fig,axes
    
def closestPointInParabola(parabola, point):   #Parabola: y=ax^2+bx+c      
    a = parabola.a
    b = parabola.b
    c = parabola.c
    
    x = point.x
    y = point.y
    
    #self.sensorOutput[0] = self.motor_command[1] #Simply takes the value of the other motor command
    
    coeff = [2.0 * a**2, 3.0 * a * b, b**2 + 2 * a * (c - y) + 1, b * (c - y) - x ]
    
    x_vals = np.real(np.roots(coeff))
    d_tmp = np.finfo(np.float64).max
    for i in range(len(x_vals)):
        x_val = x_vals[i]
        y_val = a * x_val**2 + b * x_val + c
        distance = math.sqrt( (x - x_val)** 2 + (y - y_val)** 2)
        if distance < d_tmp:
            d_tmp = distance
            x = x_val
            y = y_val
    return x, y    
        
def closestPointInCircle(circle, point):
    r = circle.r
    x_c = circle.x_c
    y_c = circle.y_c
    
    x = point.x
    y = point.y
    
    #-------------------------- r0 = math.sqrt( (x - x_c) ** 2 + (y - y_c) ** 2)
    #----------------------------------------- theta = math.acos((x - x_c) / r0)
    theta = math.atan2(y - y_c, x - x_c )
    x_p = x_c + r * math.cos(theta)
    y_p = y_c + r * math.sin(theta)  
    return x_p, y_p

def checkLineCondition(line, point):
    y_0 = line.y_0
    m = line.m
    
    x = point.x
    y = point.y
    
    if (y_0 + m * x) > y:
        return -1                # line is 'up'
    elif (y_0 + m * x) < y:
        return 1                 # line is 'down'
    elif (y_0 + m * x) == y:
        return 0                 # 'on'
    
def closestPointToLine(line, point):
    y_0 = line.y_0
    m = line.m
    
    x = point.x
    y = point.y   
            
    #-------------------------------------------------- y_0p = y + (1.0 / m) * x
    
    #------------------------------------------ x_l = (y_0p-y_0) / (m + (1.0/m))
    #x_l = (y_0p - y_0 - (1.0 / m) * x) / m
    
    
    #-------------------------------------------- x_l = (y_0p-y_0)/(m + (1.0/m))
    
    x_l = ((1/m) * x + y - y_0) / (m + (1.0/m))
    y_l = y_0 + m * x_l
    
    distance = math.sqrt( (x - x_l)** 2 + (y - y_l)** 2)

    return x_l, y_l, distance
            