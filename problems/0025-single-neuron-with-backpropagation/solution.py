import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    features_t = torch.as_tensor(features, dtype = torch.float32)
    labels_t = torch.as_tensor(labels, dtype = torch.float32)
    
    # as_tensor() has no prameter of requires_grad
    weights_t = torch.as_tensor(initial_weights , dtype = torch.float32).clone().detach().requires_grad_(True)
    bias_t = torch.tensor(initial_bias, dtype = torch.float32, requires_grad = True) 

    loss_history = []

    for _ in range(epochs):
        # forward prop
        z = torch.matmul(features_t , weights_t) + bias_t
        a = torch.sigmoid(z)
        mse_loss = torch.mean((a - labels_t)**2)
        loss_history.append(mse_loss.item())

        # backward_prop 
        mse_loss.backward()

        # weights update 
        with torch.no_grad():
            weights_t -= learning_rate*weights_t.grad
            bias_t -= learning_rate*bias_t.grad
            # clear out the gradients for next step.
            weights_t.grad.zero_()
            bias_t.grad.zero_()
    
    updated_weights = [round(w,4) for w in weights_t.tolist()]
    updated_bias = round(bias_t.item(),4)

    return (updated_weights , updated_bias , loss_history)


