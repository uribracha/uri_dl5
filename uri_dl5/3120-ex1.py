import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

array = np.array([ [[0,1,2],[10,11,12],[20,21,22]],[[100,101,102],[110,111,112],[120,121,122]]])
flat_array = array.flatten('F')
reshape_array = array.reshape(-1)
print ("flat shape: " + str(flat_array))
print ("reshape array: " + str(reshape_array))
print( time.time())