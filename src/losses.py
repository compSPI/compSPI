"""Losses."""

import torch
import torch.nn
from torch.nn import functional as F


# TODO(nina): Average on intensities, instead of sum.


def bce_on_intensities(x, recon_x, scale_b):
    """
    BCE summed over the voxels intensities.
    scale_b: plays role of loss' weighting factor.
    """
    bce = torch.sum(
        F.binary_cross_entropy(recon_x, x) / scale_b.exp() + 2 * scale_b)
    return bce


def mse_on_intensities(x, recon_x, scale_b):
    """
    MSE summed over the voxels intensities.
    scale_b: plays role of loss' weighting factor.
    """
    print(min(recon_x))
    print(max(recon_x))
    print(min(x))
    print(max(x))
    mse = F.mse_loss(recon_x, x, reduction='sum') / scale_b
    return mse


def mse_on_features(feature, recon_feature, logvar):
    """
    MSE over features of FC layer of Discriminator.
    sigma2: plays role of loss' weighting factor.
    """
    mse = F.mse_loss(recon_feature, feature) / (2 * logvar.exp())
    mse = torch.mean(mse)
    return mse


def kullback_leibler(mu, logvar):
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


def kullback_leibler_circle(mu, logvar):
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    mu_circle = mu[:, :2]
    logvar_circle = logvar[:, :2]
    mu_else = mu[:, 2:]
    logvar_else = logvar[:, 2:]

    kld_circle_attractive = -0.5 * torch.sum(
            1 + logvar_circle - mu_circle.pow(2) - logvar_circle.exp())
    kld_circle_repulsive = 0.5 * torch.sum(
            1 + logvar_circle - mu_circle.pow(2) - logvar_circle.exp())

    kld_else = -0.5 * torch.sum(
            1 + (logvar_else - (-0.6))
            - mu_else.pow(2) / (0.5**2) - logvar_else.exp() / 0.5**2)

    kld = kld_circle_attractive + kld_circle_repulsive + kld_else
    return kld


def on_circle(mu, logvar):
    mu_circle = mu[:, :2]

    on_circle = 1000 * (torch.sum(mu_circle**2, dim=1) - 1) ** 2
    on_circle = torch.sum(on_circle)
    on_circle = on_circle - 0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp())
    return on_circle


def vae_loss(x, recon_x, scale_b, mu, logvar):
    bce = bce_on_intensities(x, recon_x, scale_b)
    kld = kullback_leibler(mu, logvar)
    return bce + kld
