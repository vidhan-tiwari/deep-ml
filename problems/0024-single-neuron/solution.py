import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE
    features_t = torch.as_tensor(features , dtype = torch.float32)
    labels_t = torch.as_tensor(labels , dtype = torch.float32)
    weights_t = torch.as_tensor(weights, dtype = torch.float32)
    bias_t = torch.as_tensor(bias , dtype = torch.float32)

    z = torch.matmul(features_t , weights_t.T) + bias_t
    a = torch.sigmoid(z)
    a_list = a.tolist()
    mse = round(float(F.mse_loss(labels_t, a)),4)
    return (a_list, mse)
