"""Test functions for distributions."""
import torch
from ..src.distributions import uniform_to_triangular


def test_uniform_to_triangular():
    """Test if the distribution obtained is triangular
    We first compute the histogram obtained from this function and then
    compare it with the true histogram of a triangular distribution."""

    num_samples=1000000
    num_bins=200
    min_val=-2.0
    max_val=2.0
    bin_length=(max_val-min_val)/num_bins

    uniform_samples=torch.rand(1000000)
    triangular_samples=uniform_to_triangular(uniform_samples)
    histogram=torch.histc(triangular_samples, min=min_val, max=max_val, bins=num_bins )

    grid=torch.linspace(min_val,max_val, num_bins)
    triangle_pdf=torch.clamp(1-grid.abs(),min=0)
    true_histogram=triangle_pdf*bin_length* num_samples

    error=(true_histogram-histogram).abs().sum()/true_histogram.abs().sum()
    assert error<0.02