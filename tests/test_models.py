"""Test functions for models."""

import torch
import torch.nn as nn

import compSPI.models as models


def test_init_weights_normal():
    """Unit test for init_weights_normal"""
    dim = 16
    models.init_weights_normal(nn.Linear(dim, dim))


def test_AddCoordinates():
    """Unit test for AddCoordinates"""
    B = 5
    D = 16
    img = torch.rand(B, 1, D, D)
    layer_3 = models.AddCoordinates(with_r=False)
    layer_4 = models.AddCoordinates(with_r=True)
    assert layer_3(img).shape == (B, 3, D, D)
    assert layer_4(img).shape == (B, 4, D, D)


def test_ResidLinear():
    """Unit test for ResidLinear"""
    B = 5
    dim = 32
    x = torch.rand(B, dim)
    layer = models.ResidLinear(dim)
    layer_normal = models.ResidLinear(dim, init='normal')
    assert layer(x).shape == (B, dim)
    assert layer_normal(x).shape == (B, dim)


def test_ResidLinearMLP():
    """Unit test for ResidLinearMLP"""
    B = 5
    in_dim = 128
    nlayers = 3
    hidden_dim = 256
    out_dim = 128
    x = torch.rand(B, in_dim)
    main = models.ResidLinearMLP(in_dim, nlayers, hidden_dim, out_dim,
                                 batchnorm=True, init='normal')
    assert main(x).shape == (B, out_dim)


def test_EncoderCryoAI():
    """Unit test for EncoderCryoAI"""
    B = 5
    size = 64
    img = torch.rand(B, 1, size, size)
    mask = (torch.rand(size, size) > 0.5)
    encoder_cryoAI = models.EncoderCryoAI(size)
    encoder_cryoAI_mask = models.EncoderCryoAI(size, mask=mask, coord_conv=False)
    rot, trans = encoder_cryoAI(img)
    assert rot.shape == (B, 6)
    assert trans.shape == (B, 2)
    rot, trans = encoder_cryoAI_mask(img)
    assert rot.shape == (B, 6)
    assert trans.shape == (B, 2)