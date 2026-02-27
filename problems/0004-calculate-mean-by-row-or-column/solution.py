import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:

    a_t = torch.as_tensor(matrix,dtype = torch.float)
    if a_t.numel() == 0:
        raise ValueError("Matrix can not be empty")
    if mode not in ('row','column'):
        raise ValueError('Mode is not defined')
    if mode == 'row':
        means = torch.mean(a_t, dim = 1)
    else:
        means = torch.mean(a_t, dim = 0)
    return means

