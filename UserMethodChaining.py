import numpy as np                                                                                     
from functools import reduce                                                                           

"""
William Murphy
4/19/2018

# --- create pandas like method chaining
 using the self.__class__() dunder method

"""
                                                                                                       
class ChainedCls(object):                                                                              
    def __init__(self, calculation=0.0):                                                               
        self._calculation = result                                                                     
                                                                                                       
    def add(self, value):                                                                              
        return self.__class__(self._calculation + value)                                               
                                                                                                       
    def subtract(self, value):                                                                         
        return self.__class__(self._calculation - value)                                               
                                                                                                       
    def array_sum(self, *args):                                                                        
        return self.__class__(self._calculation + np.sum(np.array([a for a in args]).flatten()))       
                                                                                                       
    def array_mult(self, *args):                                                                       
        return self.__class__(self._calculation * reduce((lambda x, y: x*y), [a for a in args]))       
                                                                                                       
                                                                                                       
                                                                                                       
                                                                                                       
                                                                                                       
chaining = ChainedCls()                                                                                
result = (operator.add(2)                                                                              
                  .add(10)                                                                             
                  .array_sum(2,2)                                                                      
                  .array_mult(2,2,2,2)                                                                 
                  .result)                                                                             
                                                                                                       
print(result)  # prints import numpy as np                                                                                     
from functools import reduce                                                                           
                                                                                                       
                                                                                                       
class ChainedCls(object):                                                                              
    def __init__(self, calculation=0.0):                                                               
        self._calculation = result                                                                     
                                                                                                       
    def add(self, value):                                                                              
        return self.__class__(self._calculation + value)                                               
                                                                                                       
    def subtract(self, value):                                                                         
        return self.__class__(self._calculation - value)                                               
                                                                                                       
    def array_sum(self, *args):                                                                        
        return self.__class__(self._calculation + np.sum(np.array([a for a in args]).flatten()))       
                                                                                                       
    def array_mult(self, *args):                                                                       
        return self.__class__(self._calculation * reduce((lambda x, y: x*y), [a for a in args]))       
                                                                                                       
                                                                                                       
                                                                                                       
                                                                                                       
                                                                                                       
chaining = ChainedCls()                                                                                
result = (operator.add(2)                                                                              
                  .add(10)                                                                             
                  .array_sum(2,2)                                                                      
                  .array_mult(2,2,2,2)                                                                 
                  .result)                                                                             
                                                                                                       
print(result)     # prints 256                                                                                                                                                                             
