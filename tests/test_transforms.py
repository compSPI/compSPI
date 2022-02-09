"""Test functions for transforms."""

import torch
from ..src.transforms import primal_to_fourier_2D, fourier_to_primal_2D


def test_primal_to_fourier_2D():
    """Test if the fourier transform is correct.
    For this we check if the fourier transform of delta and constant images are
    constant and delta, respectively."""

    for size in [50]:

        im=torch.zeros(2, 1, size,size)
        im[0,0,size//2,size//2]=1
        im[1,0,:,:]=1

        im_fourier=primal_to_fourier_2D(im)

        expected_im_fourier=torch.zeros(2,1,size,size)
        expected_im_fourier[0]=im[1]*im[0].sum()
        expected_im_fourier[1] = im[0]*im[1].sum()

        error=(expected_im_fourier-im_fourier).abs().sum()/expected_im_fourier.abs().sum()

        assert (error<0.01)

def test_fourier_to_primal_2D():
    """Test if the inverse fourier transform is correct.
    For this we check if the fourier transform of delta and constant images are
    constant and delta, respectively."""

    for size in [50]:
        im_fourier=torch.zeros(2, 1, size,size)
        im_fourier[0,0,size//2,size//2]=1
        im_fourier[1,0,:,:]=1

        im=fourier_to_primal_2D(im_fourier)

        expected_im=torch.zeros(2,1,size,size)
        expected_im[0]=im_fourier[1]/size**2
        expected_im[1] = im_fourier[0]

        error=(expected_im-im).abs().sum()/expected_im.abs().sum()
        assert (error<0.01)