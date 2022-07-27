import torch
import numpy as np
import itertools


"""
- Inverse of BispectrumTranslation2D
- 2D Rotational Fourier Transform

"""


def fourier_so2_s1(x):
    """
    The 1D Translation Equivariant Fourier Transform

    x is a function from S1 to R

    This Fourier tranform is equivariant to the action of SO(2) on the domain.
    """
    return torch.fft(x)


def fourier_so2_s1xrplus(x):
    """
    The 2D Rotation Equivariant Fourier Transform

    x is a function from SO(2) to S1 x R+

    This Fourier tranform is equivariant to the action of SO(2) on the domain.
    """
    return


def fourier_so2xso2_s1xs1(x):
    """
    The 2D Translation Equivariant Fourier Transform

    x is a function from S1 x S1 to R

    This Fourier tranform is equivariant to the action of SO(2) x SO(2) on the domain.
    """
    return torch.fft.fft2(x)


def fourier_so3_s2(x, max_l, max_m, discretization_method):
    """
    The 3D Rotation Equivariant Fourier Transform

    x is a function from S2 to R.

    This Fourier tranform is equivariant to the action of SO(3) on the domain.
    """
    return

#
# class FourierS1xS1(torch.nn.Module):
#
#     def
#
# class FourierSO2(torch.nn.Module):

class BispectrumS1xS1(torch.nn.Module):

    """
    The 2D translation-invariant bispectrum.

    """
    def __init__(self, reduction_method="upper_triangular"):
        self.reduction_method = reduction_method

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.tensor, shape = (..., m, n)
            x is a real signal defined on S1 x S1, with option batch dimension, and (m, n) the image size

        Returns
        -------
        B: torch.tensor, dtype=torch.complex64 or torch.complex128, shape=(b, m * n * m * n)
        """
        if len(x.shape) != 3:
            x = torch.unsqueeze(x, 0)
        m, n = x.shape[-2:]
        coords = list(itertools.product(range(m), range(n)))
        X = torch.fft.fft2(x)
        t = torch.stack([torch.roll(X, (m - idxs[0], n - idxs[1]), dims=(1, 2)) for idxs in coords])
        t = torch.swapaxes(t, 0, 1)
        t = t.reshape((t.shape[0], t.shape[1], -1))
        X = self.reduce(X)
        X = X.reshape((X.shape[0], -1))
        X = torch.unsqueeze(X, -1)
        B = (torch.matmul(X, torch.swapaxes(X, 1, -1)) * torch.conj(t))
        return B

    def reduce(self, B):
        m, n = B.shape[-2:]
        if self.reduction_method == "upper_triangular":
            idx = torch.triu_indices(m, n, offset=0)
            B = B[idx]
            return B
        if self.reduction_method == "complete":
            return NotImplementedError
        if self.reduction_method is None:
            return B
        else:
            raise ValueError("Invalid reduction method {}".format(self.reduction_method))

    def inverse(self, x):
        return NotImplementedError


class BispectrumSO2(torch.nn.Module):

    """
    The 2D rotation-invariant bispectrum.

    """

    def forward(self, x):
        """
        Mathematically, we model x as a signal defined on the disk. Computationally,
        this class considers x as a signal on the 2D grid (i.e. an image).

        Parameters
        ----------
        x: torch.tensor, shape = (..., m, n)
            x is a real signal defined on the 2D grid, with option batch dimension, and (m, n) the image size

        Returns
        -------
        B: torch.tensor, dtype=torch.complex64 or torch.complex128, shape=(b, m * n * m * n)
        """
        return NotImplementedError

    def inverse(self, x):
        return NotImplementedError
