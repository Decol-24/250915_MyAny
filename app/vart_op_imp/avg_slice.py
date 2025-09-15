import numpy as np

class avg_slice:
    def __init__(self, op):
        pass

    def calculate(self, input, output):

        np_output = np.array(output, copy=False)
        np_input = np.array(input[0], copy=False)
  
        np.copyto(np_output, np_input[:,:,::2,::2])