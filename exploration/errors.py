"""
Created on: Nov 20, 2017

@uthor = Juan Manuel Acevedo Valle
"""
class ImplementationError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(ImplementationError, self).__init__(message)