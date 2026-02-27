import torch

def transpose_matrix(a) -> torch.Tensor:
    a_t = torch.as_tensor(a)
    return a_t.T