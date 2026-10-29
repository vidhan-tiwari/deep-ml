import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# find the max value 
	x = np.asarray(scores, dtype = float)
	x_max = np.max(x, axis = -1 , keepdims = True )
	shifted_x = x - x_max

	log_sum_exp = np.log(np.sum(np.exp(shifted_x) , axis = -1 , keepdims = True))

	return shifted_x - log_sum_exp