import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	features_arr = np.asarray(features, dtype = float)
	labels_arr = np.asarray(labels, dtype = float)
	weights_arr = np.asarray(weights , dtype = float)
	bias_arr = np.asarray(bias, dtype = float)

	z = features_arr @ weights_arr.T + bias_arr

	probabilities = 1.0/(1.0 + np.exp(-z))

	mse = np.mean((labels_arr - probabilities)**2)

	return probabilities.tolist(), round(float(mse),4)