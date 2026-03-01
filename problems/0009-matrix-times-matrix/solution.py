import torch 

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    a_t = torch.as_tensor(a ,dtype = torch.float32)
    b_t = torch.as_tensor(b, dtype = torch.float32)
    if a_t.ndim <2 or b_t.ndim <2:
        return torch.as_tensor(-1)
    if a_t.shape[1] != b_t.shape[0]:
        return torch.as_tensor(-1)
    return a_t @ b_t
