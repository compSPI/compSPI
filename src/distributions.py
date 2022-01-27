"""contain functions dealing with probability distributions."""

def uniform_to_triangular(x):
    """Convert samples from uniform distribution to triangular distribution.
    Parameters
    ----------
    x: torch.Tensor
    Returns
    -------
    out: torch.Tensor
    """
    return (x - 0.5).sign() * (1 - (1 - (2 * x - 1).abs()).sqrt())

