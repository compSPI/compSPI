"""contain functions dealing with image transformations."""
import torch


def primal_to_fourier_2D(r):
    """Return 2D fourier transform of a batch of images.

    Parameters
    ----------
    r: torch.Tensor
        Tensor of size (batch, 1, size, size)

    Returns
    -------
    out: torch.Tensor
        Tensor of size (batch, 1, size, size)
    """
    r = torch.fft.fftshift(r, dim=(-2, -1))
    return torch.fft.ifftshift(
        torch.fft.fftn(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1)), dim=(-2, -1)
    )


def fourier_to_primal_2D(f):
    """Return inverse 2D fourier transform of a batch of images.

    Parameters
    ----------
    f: torch.Tensor
        Tensor of size (batch,1, size,size)

    Returns
    -------
    out: torch.Tensor
        Tensor of size (batch,1, size,size)
    """
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    return torch.fft.fftshift(
        torch.fft.ifftn(f, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1)), dim=(-2, -1)
    )


def primal_to_fourier_3D(r):
    """Return 3D fourier transform of a batch of images.

    Parameters
    ----------
    r: torch.Tensor
        Tensor of size (batch, size, size, size)
    Returns
    -------
    out: torch.Tensor
        Tensor of size (batch, size, size, size)
    """
    r = torch.fft.fftshift(r, dim=(-3, -2, -1))
    return torch.fft.ifftshift(
        torch.fft.fftn(r, s=(r.shape[-3], r.shape[-2], r.shape[-1]), dim=(-3, 2, -1)),
        dim=(-3, -2, -1),
    )


def fourier_to_primal_3D(f):
    """Return inverse 3D fourier transform of a batch of images.

    Parameters
    ----------
    f: torch.Tensor
        Tensor of size (batch, size, size, size)
    Returns
    -------
    out: torch.Tensor
        Tensor of size (batch, size, size, size)
    """
    f = torch.fft.ifftshift(f, dim=(-3, -2, -1))
    return torch.fft.fftshift(
        torch.fft.ifftn(f, s=(f.shape[-3], f.shape[-2], f.shape[-1]), dim=(-3, -2, -1)),
        dim=(-3, -2, -1),
    )
