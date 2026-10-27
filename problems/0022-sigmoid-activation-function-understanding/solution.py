import math
import numpy as np
def sigmoid(z: float) -> float:
	#Your code here
	result = 1.0 /(1.0 + np.exp(-z))
	return result