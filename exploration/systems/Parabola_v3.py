'''
Created on Feb 5, 2016
This sensorimor system defines the DIVA agent used for the CCIA 2015's paper
@author: Juan Manuel Acevedo Valle
'''

# import sys
# import wave
import math
import numpy as np
from matplotlib import pyplot as plt


# from matplotlib.pyplot import autoscale
# from matplotlib.animation import Animation
# from scipy.interpolate.interpolate_wrapper import block
class CustomObject:
    def __init__(self):
        pass


class ParabolicRegion:
    def __init__(self, sigma_noise=0.0000001):

        f = 0.

        a = 2.0 + f
        b = 3.0
        c = 3.0 + f
        d = 3.5 + f
        e = 1.5 + f

        m1 = 1
        m2 = 1

        motor_names = ['M1', 'M2']
        somato_names = ['S1', 'S2']
        sensor_names = ['SS1','SS2','SS3','SS4']
        cons_names = ['P1']
        n_motor = 2
        n_somato = 2
        n_sensor = 4
        n_cons = 1

        min_motor_values = np.array([0.0, 0.0])
        max_motor_values = np.array([b * 2.0, b * 2.0])

        min_somato_values = np.array([0.0, 0.0 + f])
        max_somato_values = np.array([b * 2.0, b ** 2 + f])

        min_sensor_values = np.array([0.0, 0.0, 0.0, 0.0])
        max_sensor_values = np.array([9.0, 9.0, 9.0, 9.0])

        min_cons_values = np.array([0])
        max_cons_values = np.array([1])
        cons_threshold = np.array([0.6])

        name = 'ParabolicRegion'
        self.name = name

        self.params = CustomObject()
        self.params.a = a
        self.params.b = b
        self.params.c = c
        self.params.d = d
        self.params.e = e
        self.params.f = f

        self.params.m1 = m1
        self.params.m2 = m2
        self.params.sigma_noise = sigma_noise

        self.n_motor = n_motor
        self.n_somato = n_somato
        self.n_sensor = n_sensor
        self.n_cons = n_cons
        self.motor_names = motor_names
        self.somato_names = somato_names
        self.cons_names = cons_names
        self.sensor_names = sensor_names

        self.min_motor_values = min_motor_values
        self.max_motor_values = max_motor_values
        self.min_somato_values = min_somato_values
        self.max_somato_values = max_somato_values
        self.min_sensor_values = min_sensor_values
        self.max_sensor_values = max_sensor_values
        self.min_cons_values = min_cons_values
        self.max_cons_values = max_cons_values
        self.cons_threshold = cons_threshold

        self.motor_command = np.array([0.0] * n_motor)
        self.somato_out = np.array([0.0] * n_somato)
        self.somato_goal = np.array([0.0] * n_somato)
        self.sensor_goal = np.array([0.0] * n_sensor)
        self.sensor_out = np.array([0.0] * n_sensor)
        self.cons_out = np.array([0.0] * n_cons)
        self.competence_result = 0.0
        self.sensor_instructor = np.empty((self.n_sensor,))
        self.sensor_instructor.fill(np.nan)


    def set_f(self, f):
        a = 2.0 + f
        b = 3.0
        c = 3.0 + f
        d = 3.5 + f
        e = 1.5 + f
        self.params.a = a
        self.params.b = b
        self.params.c = c
        self.params.d = d
        self.params.e = e
        self.params.f = f

        self.min_somato_values = np.array([0.0, 0.0 + f])
        self.max_somato_values = np.array([b * 2.0, b ** 2 + f])


    def set_action(self, motor_command):
        self.motor_command = self.boundMotorCommand(motor_command)

    def execute_action_unconstrained(self):
        self.motor_command = self.boundMotorCommand(self.motor_command)

        a = self.params.a
        b = self.params.b
        c = self.params.c

        self.cons_out = 0.0
        self.somato_out[0] = self.motor_command[0]
        self.somato_out[1] = math.pow(self.motor_command[1] - b, 2.0) + self.params.f

    def execute_action(self):
        self.motor_command = self.boundMotorCommand(self.motor_command)

        self.execute_action_unconstrained()
        self.applyConstraints()

        self.somato_out[0] = self.somato_out[0] + np.random.normal(0.0, self.params.sigma_noise, 1)
        self.somato_out[1] = self.somato_out[1] + np.random.normal(0.0, self.params.sigma_noise, 1)

        self.applyConstraints()

        # SOMATO OUT
        a = self.params.a
        b = self.params.b
        c = self.params.c
        d = self.params.d
        e = self.params.e
        f = self.params.f

        m1 = self.params.m1
        m2 = self.params.m2

        r = c - a
        point = CustomObject()
        point.x = self.somato_out[0]
        point.y = self.somato_out[1]

        # Checking inner Parabolic Region Condition
        parabola = CustomObject()
        parabola.a = 1.0
        parabola.b = -2.0 * b
        parabola.c = b ** 2 + f

        ##Checking if the somatoimotor result is insede of the constrained circle
        circle = CustomObject()
        circle.x_c = b
        circle.y_c = a
        circle.r = r

        ## Checking if the somatoimotor result is inside of thecontrained region between two parallel lines
        up_line = CustomObject()
        up_line.y_0 = d
        up_line.m = m1

        down_line = CustomObject()
        down_line.y_0 = e
        down_line.m = m2

        foo1, foo2, som1 = closestPointInParabola(parabola=parabola, point=point)
        foo1, foo2, som2 = closestPointInCircle(circle=circle, point=point)
        foo1, foo2, som3 = closestPointToLine(line=down_line, point=point)
        foo1, foo2, som4 = closestPointToLine(line=up_line, point=point)
        self.sensor_out[0] = som1
        self.sensor_out[1] = som2
        self.sensor_out[2] = som3
        self.sensor_out[3] = som4


        # ---------------------------------------- while self.applyConstraints():
        # -------------------------------------------------------------- pass

    def applyConstraints(self):
        a = self.params.a
        b = self.params.b
        c = self.params.c
        d = self.params.d
        e = self.params.e
        f = self.params.f

        m1 = self.params.m1
        m2 = self.params.m2

        r = c - a
        x = self.somato_out[0]
        y = self.somato_out[1]

        changed = False
        onParabola = False
        point = CustomObject()
        point.x = x
        point.y = y

        # Checking inner Parabolic Region Condition
        parabola = CustomObject()
        parabola.a = 1.0
        parabola.b = -2.0 * b
        parabola.c = b ** 2 + f

        ##Checking if the somatoimotor result is insede of the constrained circle
        circle = CustomObject()
        circle.x_c = b
        circle.y_c = a
        circle.r = r
        circle_condition = math.pow(x - b, 2.0) + math.pow(y - a, 2.0)

        ## Checking if the somatoimotor result is inside of thecontrained region between two parallel lines
        up_line = CustomObject()
        up_line.y_0 = d
        up_line.m = m1

        down_line = CustomObject()
        down_line.y_0 = e
        down_line.m = m2

        uppest_line = CustomObject()
        uppest_line.y_0 = self.max_somato_values[1]+f
        uppest_line.m = 0.0

        if circle_condition < math.pow(r, 2.0):  # Circle
            changed = True
            self.cons_out = 1.0
            x, y, foo = closestPointInCircle(circle, point)
            point.x = x
            point.y = y

        # if math.pow(self.motor_command[0]-b,2.0) > self.somato_out[1]: #Parabola
        if math.pow(self.somato_out[0] - b, 2.0) + f > self.somato_out[1]:  # Parabola
            changed = True
            onParabola = True
            self.cons_out = 1.0
            x, y, foo = closestPointInParabola(parabola, point)
            point.x = x
            point.y = y

        if (checkLineCondition(up_line, point) == -1 and checkLineCondition(down_line, point) == 1):  # Lines
            changed = True
            self.cons_out = 1.0
            x1, y1, distance1 = closestPointToLine(up_line, point)
            x2, y2, distance2 = closestPointToLine(down_line, point)

            x = point.x
            y = point.y
            if onParabola:
                if distance1 >= distance2:
                    x_vals, y_vals = intersectionParabolaLine(parabola, down_line)
                    d_tmp = np.finfo(np.float64).max

                    x_tmp = x
                    y_tmp = y
                    for i in range(len(x_vals)):
                        x_val = x_vals[i]
                        y_val = y_vals[i]
                        distance = math.sqrt((x_tmp - x_val) ** 2 + (y_tmp - y_val) ** 2)
                        if distance < d_tmp:
                            d_tmp = distance
                            x = x_val
                            y = y_val
                else:
                    x_vals, y_vals = intersectionParabolaLine(parabola, up_line)
                    d_tmp = np.finfo(np.float64).max
                    for i in range(len(x_vals)):
                        x_val = x_vals[i]
                        y_val = y_vals[i]
                        distance = math.sqrt((x - x_val) ** 2 + (y - y_val) ** 2)
                        if distance < d_tmp:
                            d_tmp = distance
                            x = x_val
                            y = y_val
            else:
                if distance1 >= distance2:
                    x = x2
                    y = y2
                else:
                    x = x1
                    y = y1

            point.x = x
            point.y = y

        if checkLineCondition(uppest_line, point) == 1:  # Upper limit
            changed = True
            self.cons_out = 1.0
            x, y, distance = closestPointToLine(uppest_line, point)
            point.x = x
            point.y = y
            if checkLineCondition(up_line, point) == -1:
                x, y = intersectionTwoLines(uppest_line, up_line)
                point.x = x
                point.y = y
            elif onParabola:
                x, y, foo = closestPointInParabola(parabola, point)
                point.x = x
                point.y = y

        self.somato_out[0] = x
        self.somato_out[1] = y
        return changed

    def boundMotorCommand(self, motor_command):
        n_motor = self.n_motor
        min_motor_values = self.min_motor_values
        max_motor_values = self.max_motor_values

        for i in range(n_motor):
            if (motor_command[i] < min_motor_values[i]):
                motor_command[i] = min_motor_values[i]

            elif (motor_command[i] > max_motor_values[i]):
                motor_command[i] = max_motor_values[i]

        return motor_command

    def drawSystem(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1,1)
        plt.sca(axes)

        min_somato_values = self.min_somato_values
        max_somato_values = self.max_somato_values

        a = self.params.a
        b = self.params.b
        c = self.params.c
        d = self.params.d
        e = self.params.e
        f = self.params.f
        m1 = self.params.m1
        m2 = self.params.m2

        r = c - a

        x_p = np.linspace(min_somato_values[0], max_somato_values[0], 100)
        y_p = (x_p - b) ** 2 +f

        circle = plt.Circle((b, a), r, color='red')

        x_l1 = np.linspace(min_somato_values[0], max_somato_values[0], 100)
        y_l1 = d + m1 * x_l1

        x_l2 = np.linspace(min_somato_values[0], max_somato_values[0], 100)
        y_l2 = e + m2 * x_l2

        plt.sca(axes)

        plt.plot(x_l1, y_l1, "--r")
        plt.plot(x_l2, y_l2, "--r")
        gap = d - e
        for i in range(100):
            plt.plot(x_l2, y_l2 + i * gap / 100, "r")

        axes.add_artist(circle)
        axes.set_xlim([-3.0, 9.0])
        axes.set_ylim([-1.0, 11.0])

        plt.plot(x_p, y_p, "b")

class Instructor(ParabolicRegion):
    def __init__(self):
        import os
        from ..data.data import load_sim_h5
        # ss = [[3., 0.5], [1.9, 1.25], [4.15, 2],[2.3, 3.4], [5.27, 6.23], [0.15, 8.7], [2.36, 7.46], [5.2, 8.87]]
        #     for s in ss:
        #         system.set_action(infer_motor(0, s))
        #         system.execute_action()
        #         data.append_data(system)
        ParabolicRegion.__init__(self)
        abs_path = os.path.dirname(os.path.abspath(__file__))
        self.intructor_file = abs_path + '/datasets/parabola_v3_dataset.h5'
        self.data = load_sim_h5(self.intructor_file)
        self.n_units = len(self.data.sensor.data.index)
        self.unit_threshold = 0.15 # 0.3

    def plot(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1,1)
        plt.sca(axes)
        x=[]
        y=[]
        for i in range(self.n_units):
            self.set_action(self.data.motor.data.iloc[i])
            self.execute_action()
            x += [self.sensor_out[0]]
            y += [self.sensor_out[1]]
        plt.plot(x,y,'o')


    def interaction(self, sensor):
        dist = np.array(self.get_distances(sensor))
        min_idx = np.argmin(dist)
        self.min_idx = min_idx
        if dist[min_idx] <= self.unit_threshold:
            self.set_action(self.data.motor.data.iloc[min_idx])
            self.execute_action()
            return 1, self.sensor_out.copy()
        return 0, np.zeros((self.n_sensor,))


    def get_distances(self, sensor):
        dist = []
        s_data = self.data.sensor.data.as_matrix()
        for i in range(self.n_units):
            dist += [np.linalg.norm(sensor - s_data[i, :])]
        return dist

    def infer_motor(self, sensor_goal):
        m1 = sensor_goal[0]
        m2 = np.sqrt(sensor_goal[1])+3
        return np.array([m1, m2])


def closestPointInParabola(parabola, point):  # Parabola: y=ax^2+bx+c
    a = parabola.a
    b = parabola.b
    c = parabola.c

    x = point.x
    y = point.y

    # self.somato_out[0] = self.motor_command[1] #Simply takes the value of the other art command

    coeff = [2.0 * a ** 2, 3.0 * a * b, b ** 2 + 2 * a * (c - y) + 1, b * (c - y) - x]

    x_vals = np.real(np.roots(coeff))
    d_tmp = np.finfo(np.float64).max

    x_0 = x
    y_0 = y
    for i in range(len(x_vals)):
        x_val = x_vals[i]
        y_val = a * x_val ** 2 + b * x_val + c
        distance = math.sqrt((x_0 - x_val) ** 2 + (y_0 - y_val) ** 2)
        if distance < d_tmp:
            d_tmp = distance
            x = x_val
            y = y_val
    return x, y, d_tmp


def intersectionTwoLines(line1, line2):
    y_01 = line1.y_0
    m1 = line1.m
    y_02 = line2.y_0
    m2 = line2.m

    x = (y_02 - y_01) / (m1 - m2)
    y = m1 * x + y_01
    return x, y


def intersectionParabolaLine(parabola, line):  # Parabola: y=ax^2+bx+c
    a = parabola.a
    b = parabola.b
    c = parabola.c

    y_0 = line.y_0
    m = line.m

    # self.somato_out[0] = self.motor_command[1] #Simply takes the value of the other art command

    coeff = [a, b - m, c - y_0]

    x_vals = np.real(np.roots(coeff))
    y_vals = [0.0, 0.0]
    for i in range(len(x_vals)):
        x_val = x_vals[i]
        y_vals[i] = a * x_val ** 2 + b * x_val + c
    return x_vals, y_vals


def closestPointInCircle(circle, point):
    r = circle.r
    x_c = circle.x_c
    y_c = circle.y_c

    x = point.x
    y = point.y

    # -------------------------- r0 = math.sqrt( (x - x_c) ** 2 + (y - y_c) ** 2)
    # ----------------------------------------- theta = math.acos((x - x_c) / r0)
    theta = math.atan2(y - y_c, x - x_c)
    x_p = x_c + r * math.cos(theta)
    y_p = y_c + r * math.sin(theta)
    distance = math.sqrt((x - x_p) ** 2 + (y - y_p) ** 2)
    return x_p, y_p, distance


def checkLineCondition(line, point):
    y_0 = line.y_0
    m = line.m

    x = point.x
    y = point.y

    if (y_0 + m * x) > y:
        return -1  # line is 'up'
    elif (y_0 + m * x) < y:
        return 1  # line is 'down'
    elif (y_0 + m * x) == y:
        return 0  # 'on'


def closestPointToLine(line, point):
    y_0 = line.y_0

    m = line.m

    x = point.x
    y = point.y

    if m != 0:

        # -------------------------------------------------- y_0p = y + (1.0 / m) * x

        # ------------------------------------------ x_l = (y_0p-y_0) / (m + (1.0/m))
        # x_l = (y_0p - y_0 - (1.0 / m) * x) / m


        # -------------------------------------------- x_l = (y_0p-y_0)/(m + (1.0/m))

        x_l = ((1 / m) * x + y - y_0) / (m + (1.0 / m))
        y_l = y_0 + m * x_l
    else:
        x_l = x
        y_l = y_0

    distance = math.sqrt((x - x_l) ** 2 + (y - y_l) ** 2)

    return x_l, y_l, distance
