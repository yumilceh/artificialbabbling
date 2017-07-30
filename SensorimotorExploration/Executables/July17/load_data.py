from SensorimotorExploration.DataManager.SimulationData import load_sim_h5_v2 as load_sim_h5
data_file = 'epirob/Vowels_Tree_0_2017_03_28_02_40_sim_data.h5'
data, foo = load_sim_h5(data_file)

motor_data = data.motor.data.as_matrix()
sensor_data = data.sensor.data.as_matrix()
somato_data = data.somato.data.as_matrix()