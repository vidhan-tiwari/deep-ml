import numpy as np

def train_neuron(
    features: np.ndarray, 
    labels: np.ndarray, 
    initial_weights: np.ndarray, 
    initial_bias: float, 
    learning_rate: float, 
    epochs: int
) -> tuple[np.ndarray, float, list[float]]:
    
    features_ar = np.asarray(features, dtype=float)
    labels_ar = np.asarray(labels, dtype=float)
    weights_ar = np.asarray(initial_weights, dtype=float)
    bias_ar = float(initial_bias)
    
    n_samples = len(features_ar)
    loss_history = []
    
    for _ in range(epochs):
        # 1. Forward Pass
        z = features_ar @ weights_ar + bias_ar
        a = 1.0 / (1.0 + np.exp(-z))
        mse = float(np.mean((a - labels_ar) ** 2))
        loss_history.append(round(mse, 4))
        dz = (2.0 / n_samples) * (a - labels_ar) * a * (1.0 - a)
        dw = features_ar.T @ dz
        db = float(np.sum(dz))

        # 4. Gradient Descent Update
        weights_ar -= learning_rate * dw
        bias_ar -= learning_rate * db

    return np.round(weights_ar, 4), round(float(bias_ar), 4), loss_history