"""Test functions for transforms."""

import torch

from compSPI.transforms import fourier_to_primal_2D as ifft_2d
from compSPI.transforms import fourier_to_primal_3D as ifft_3d
from compSPI.transforms import primal_to_fourier_2D as fft_2d
from compSPI.transforms import primal_to_fourier_3D as fft_3d


def test_primal_to_fourier_2D():
    """Test if the 2D fourier transform is correct.

    For this we check if the fourier transform of delta and constant images are
    constant and delta, respectively.
    """
    for size in [50]:

        im = torch.zeros(2, 1, size, size)
        im[0, 0, size // 2, size // 2] = 1
        im[1, 0, :, :] = 1

        im_fourier = fft_2d(im)

        expected_im_fourier = torch.zeros(2, 1, size, size)
        expected_im_fourier[0] = im[1] * im[0].sum()
        expected_im_fourier[1] = im[0] * im[1].sum()

        error = (
            expected_im_fourier - im_fourier
        ).abs().sum() / expected_im_fourier.abs().sum()

        assert error < 0.01


def test_fourier_to_primal_2D():
    """Test if the inverse 2D fourier transform is correct.

    For this we check if the fourier transform of delta and constant images are
    constant and delta, respectively.
    """
    for size in [50]:

        im_fourier = torch.zeros(2, 1, size, size)
        im_fourier[0, 0, size // 2, size // 2] = 1
        im_fourier[1, 0, :, :] = 1

        im = ifft_2d(im_fourier)

        expected_im = torch.zeros(2, 1, size, size)
        expected_im[0] = im_fourier[1] / size**2
        expected_im[1] = im_fourier[0]

        error = (expected_im - im).abs().sum() / expected_im.abs().sum()
        assert error < 0.01


def test_primal_to_fourier_3D():
    """Test if the 3D fourier transform is correct.

    For this we check if the fourier transform of delta and constant images are
    constant and delta, respectively.
    """
    for size in [50]:

        im = torch.zeros(2, size, size, size)
        im[0, size // 2, size // 2, size // 2] = 1
        im[1, :, :, :] = 1

        im_fourier = fft_3d(im)

        expected_im_fourier = torch.zeros(2, size, size, size)
        expected_im_fourier[0] = im[1] * im[0].sum()
        expected_im_fourier[1] = im[0] * im[1].sum()

        error = (
            expected_im_fourier - im_fourier
        ).abs().sum() / expected_im_fourier.abs().sum()

        assert error < 0.01


def test_fourier_to_primal_3D():
    """Test if the inverse 3D fourier transform is correct.

    For this we check if the fourier transform of delta and constant images are
    constant and delta, respectively.
    """
    for size in [50]:

        im_fourier = torch.zeros(2, size, size, size)
        im_fourier[0, size // 2, size // 2, size // 2] = 1
        im_fourier[1, :, :, :] = 1

        im = ifft_3d(im_fourier)

        expected_im = torch.zeros(2, size, size, size)
        expected_im[0] = im_fourier[1] / size**3
        expected_im[1] = im_fourier[0]

        error = (expected_im - im).abs().sum() / expected_im.abs().sum()
        assert error < 0.01
