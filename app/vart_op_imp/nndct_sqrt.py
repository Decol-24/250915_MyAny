import numpy as np

class nndct_sqrt:
    def __init__(self, op):
        pass

    def calculate(self, input, output):

        np_output = np.array(output, copy=False)
        np_input = np.array(input[0], copy=False)

        np_output = np.sqrt(np_input,out=np_output)