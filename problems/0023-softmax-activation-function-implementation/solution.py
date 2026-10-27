import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
    # Your code here
    scores_ar = np.asarray(scores, dtype = float)
    # to deal with numerical instability
    shifted_scores = scores_ar - np.max(scores_ar)

    soft_scores = np.exp(shifted_scores)/(np.sum(np.exp(shifted_scores)))
    
    return soft_scores