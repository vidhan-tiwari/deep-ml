import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    m = matrix
    a,b = m[0,0],m[0,1]
    c,d = m[1,0],m[1,1]
    trace = a+d
    det = (a*d) -(b*c)
    discriminant = trace**2 - 4*det
    sqrt_disc = torch.sqrt(discriminant)
    l1 = (trace + sqrt_disc) / 2
    l2 = (trace - sqrt_disc) / 2
    return torch.stack([l1,l2])
