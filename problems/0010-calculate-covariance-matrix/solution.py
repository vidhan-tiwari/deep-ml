import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	arr = np.array(vectors)
	means = np.mean(vectors, axis = 1,keepdims = True)
	deviations = arr - means
	return (deviations @ deviations.T)/ (arr.shape[1]-1) # number of observations 
