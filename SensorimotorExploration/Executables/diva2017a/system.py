# Import libraries
from SensorimotorExploration.Systems.Diva2017a import Diva2017a as Diva
import numpy as np
import matplotlib.pyplot as plt

# Create System
system = Diva()

# Parameters and Motor command
duration_m1 = 0.4
duration_m2 = 0.4
w0 = 2 * np.pi / 0.01
damping_factor = 1.01
motor_command = np.array([0.1]*10 + [1.]*3 + [0.2]*10 + [1.]*3)
print(motor_command)


par_kargs = {'duration_m1': duration_m1,
             'duration_m2': duration_m2,
             'w0': w0,
             'damping_factor': damping_factor}

system.set_params(**par_kargs)

system.set_action(motor_command)
system.execute_action()


fig,ax = plt.subplots(1,1)
system.plot_arts_evo(range(13),axes=ax)
plt.hold(True)
system.plot_cons_out(axes=ax)
# system.plotSoundWave()

print(system.art_states)
plt.show()