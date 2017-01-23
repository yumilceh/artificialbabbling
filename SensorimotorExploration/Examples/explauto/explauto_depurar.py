'''
Created on Jan 23, 2017

@author: Juan Manuel Acevedo Valle
'''

if __name__ == '__main__':
    from explauto.environment.environment import Environment
    environment = Environment.from_configuration('simple_arm', 'mid_dimensional')
    from explauto.sensorimotor_model import sensorimotor_models, available_configurations
    
    from explauto.sensorimotor_model.sensorimotor_model import SensorimotorModel
    sm_model = SensorimotorModel.from_configuration(environment.conf, "nearest_neighbor", "default")
        
    
    
    