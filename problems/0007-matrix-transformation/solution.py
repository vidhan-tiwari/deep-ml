import numpy as np
def is_invertible(arr : np.array) -> bool:
	if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
		return False
	return np.linalg.matrix_rank(arr) == arr.shape[0] 
def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	A_np = np.array(A)
	T_np = np.array(T)
	S_np = np.array(S)
	if not is_invertible(T_np) or not is_invertible(S_np):
		return -1
	result = 0
	if T_np.shape[0] != A_np.shape[0]:
		return -1

	result = np.linalg.inv(T_np) @ A_np
	if result.shape[1] != S_np.shape[0]:
		return -1
	result = result @ S_np
	return result

	