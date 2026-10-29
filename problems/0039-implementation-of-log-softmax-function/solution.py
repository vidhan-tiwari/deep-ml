import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    # Your code here
    x_t = torch.as_tensor(scores , dtype = torch.float32)
    max_x_t = torch.amax(x_t , dim = -1 , keepdim = True)
    shifted_x_t = x_t - max_x_t

    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_x_t), dim = -1 , keepdim = True))
    
    return shifted_x_t - log_sum_exp
