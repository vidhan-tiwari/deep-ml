import torch

def matrix_dot_vector(a, b) -> torch.Tensor:
    """
    Compute the product of matrix `a` and vector `b` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of length m, or tensor(-1) if dimensions mismatch.
    """
    a_t = torch.as_tensor(a,dtype = torch.float)
    b_t = torch.as_tensor(b,dtype = torch.float)
    # ensuring dimension compatibility 
    # a_t should be a 2d matrix and b_t should be a 1d vector 
    if a_t.dim() != 2 or b_t.dim() != 1:
        return torch.tensor(-1)
    if a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)
    return a_t @ b_t 
