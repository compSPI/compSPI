"""NN fabric."""

import importlib
import os

import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.optim
import torch.utils.data

import geomstats
from geomstats.geometry.euclidean_space import EuclideanSpace
from geomstats.geometry.hyperbolic_space import HyperbolicSpace
from geomstats.geometry.hypersphere import Hypersphere

R2 = EuclideanSpace(dimension=2)
H2 = HyperbolicSpace(dimension=2)
S2 = Hypersphere(dimension=2)

MANIFOLD = {'r2': R2, 'h2': H2, 's2': S2}

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


def make_decoder_true(synthetic_params, nn_architecture):
    w_true = synthetic_params['w_true']
    b_true = synthetic_params['b_true']
    data_dim = synthetic_params['data_dim']

    latent_dim = nn_architecture['latent_dim']
    n_layers = nn_architecture['n_decoder_layers']
    nonlinearity = nn_architecture['nonlinearity']
    with_biasx = nn_architecture['with_biasx']
    with_logvarx = nn_architecture['with_logvarx']
    logvarx_true = nn_architecture['logvarx_true']

    decoder_true = Decoder(
        latent_dim=latent_dim, data_dim=data_dim,
        n_layers=n_layers,
        nonlinearity=nonlinearity,
        with_biasx=with_biasx,
        with_logvarx=with_logvarx,
        logvarx_true=logvarx_true)
    decoder_true.to(DEVICE)

    for i in range(n_layers):
        decoder_true.layers[i].weight.data = torch.tensor(
            w_true[i]).to(DEVICE)
        if with_biasx:
            decoder_true.layers[i].bias.data = torch.tensor(
                b_true[i]).to(DEVICE)

    if with_logvarx:
        # Layer predicting logvarx
        decoder_true.layers[n_layers].weight.data = torch.tensor(
            w_true[n_layers]).to(DEVICE)
        decoder_true.layers[n_layers].bias.data = torch.tensor(
            b_true[n_layers]).to(DEVICE)

    return decoder_true


def generate_from_decoder(decoder, n_samples=1):
    z, mux, logvarx = decoder.generate(n_samples=n_samples)
    _, data_dim = mux.shape

    mux = mux.cpu().detach().numpy()
    logvarx = logvarx.cpu().detach().numpy()

    generated_x = np.zeros((n_samples, data_dim))
    for i in range(n_samples):
        logvar = logvarx[i].squeeze()
        sigma = np.sqrt(np.exp((logvar)))
        eps = np.random.normal(
            loc=0, scale=sigma, size=(1, data_dim))
        generated_x[i] = mux[i] + eps

    return generated_x


def generate_from_decoder_fixed_var(decoder, logvarx=1, n_samples=1):
    z, mux, _ = decoder.generate(n_samples=n_samples)
    _, data_dim = mux.shape

    mux = mux.cpu().detach().numpy()

    generated_x = np.zeros((n_samples, data_dim))
    for i in range(n_samples):
        logvar = logvarx
        sigma = np.sqrt(np.exp((logvar)))
        eps = np.random.normal(
            loc=0, scale=sigma, size=(1, data_dim))
        generated_x[i] = mux[i] + eps

    return generated_x


def convert_to_tangent_space(x, manifold_name='s2'):
    n_samples, _ = x.shape
    if type(x) == np.ndarray:
        if manifold_name == 's2':
            x_vector_extrinsic = np.hstack([x, np.zeros((n_samples, 1))])
        elif manifold_name == 'h2':
            x_vector_extrinsic = np.hstack([np.zeros((n_samples, 1)), x])
        elif manifold_name == 'r2':
            x_vector_extrinsic = x
        else:
            raise ValueError('Manifold not supported.')
    elif type(x) == torch.Tensor:
        if os.environ['GEOMSTATS_BACKEND'] == 'numpy':
            os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
            importlib.reload(geomstats.backend)
        if manifold_name == 's2':
            x_vector_extrinsic = torch.cat(
                [x, torch.zeros((n_samples, 1)).to(DEVICE)], dim=1)
        elif manifold_name == 'h2':
            x_vector_extrinsic = torch.cat(
                [torch.zeros((n_samples, 1)).to(DEVICE), x], dim=1)
        elif manifold_name == 'r2':
            x_vector_extrinsic = x
        else:
            raise ValueError('Manifold not supported.')

    return x_vector_extrinsic


def manifold_and_base_point(manifold_name):
    manifold = MANIFOLD[manifold_name]
    if os.environ['GEOMSTATS_BACKEND'] == 'numpy':
        if manifold_name == 's2':
            base_point = np.array([0, 0, 1])
        elif manifold_name == 'h2':
            base_point = np.array([1, 0, 0])
        elif manifold_name == 'r2':
            base_point = np.array([0, 0])
        else:
            raise ValueError('Manifold not supported.')
    elif os.environ['GEOMSTATS_BACKEND'] == 'pytorch':
        if manifold_name == 's2':
            base_point = torch.Tensor([0., 0., 1.]).to(DEVICE)
        elif manifold_name == 'h2':
            base_point = torch.Tensor([1., 0., 0.]).to(DEVICE)
        elif manifold_name == 'r2':
            base_point = torch.Tensor([0., 0.]).to(DEVICE)
        else:
            raise ValueError('Manifold not supported.')
    return manifold, base_point


def generate_from_decoder_fixed_var_riem(decoder, logvarx=1, n_samples=1,
                                         manifold_name='h2'):
    """
    The decoder generate on the tangent space of a manifold.
    We use Exp to bring these points on the manifold.
    We add a Gaussian noise at each point.
    To this aim, we use a wrapped Gaussian: we generate a Gaussian noise
    at the tangent space of the point, and use the Exp at the point to
    get a point on the manifold.
    """
    # TODO(nina): Extend to more dimensions

    z, mux, _ = decoder.generate(n_samples=n_samples)
    mux = mux.detach().cpu().numpy()
    _, data_dim = mux.shape

    mux = convert_to_tangent_space(mux, manifold_name=manifold_name)
    manifold, base_point = manifold_and_base_point(manifold_name)

    mux_riem = manifold.metric.exp(mux, base_point=base_point)

    scale = np.sqrt(np.exp(logvarx))
    eps = np.random.normal(
        loc=0, scale=scale, size=(n_samples, data_dim+1))  # HACK!
    eps = manifold.projection_to_tangent_space(
        vector=eps, base_point=mux_riem)

    generated_x = manifold.metric.exp(eps, base_point=mux_riem)

    return generated_x


def generate_from_decoder_fixed_var_tgt(decoder, logvarx=1, n_samples=1,
                                        manifold_name='h2'):
    """
    Bring the generated points back on the tangent space
    at the chosen basepoint.
    """
    # TODO(nina): Extend to more dimensions

    generated_x = generate_from_decoder_fixed_var_riem(
        decoder, logvarx, n_samples, manifold_name)

    manifold, base_point = manifold_and_base_point(manifold_name)

    generated_x_on_tangent_space = manifold.metric.log(
        generated_x, base_point=base_point)
    if manifold_name == 's2':
        generated_x_on_tangent_space = generated_x_on_tangent_space[:, :2]
    elif manifold_name == 'h2':
        generated_x_on_tangent_space = generated_x_on_tangent_space[:, 1:]
    return generated_x_on_tangent_space


def reparametrize(mu, logvar, n_samples=1):
    n_batch_data, latent_dim = mu.shape

    std = logvar.mul(0.5).exp_()
    std_expanded = std.expand(
        n_samples, n_batch_data, latent_dim)
    mu_expanded = mu.expand(
        n_samples, n_batch_data, latent_dim)

    if CUDA:
        eps = torch.cuda.FloatTensor(
            n_samples, n_batch_data, latent_dim).normal_()
    else:
        eps = torch.FloatTensor(n_samples, n_batch_data, latent_dim).normal_()
    eps = torch.autograd.Variable(eps)

    z = eps * std_expanded + mu_expanded
    z_flat = z.resize(n_samples * n_batch_data, latent_dim)
    z_flat = z_flat.squeeze()  # case where latent_dim = 1: squeeze last dim
    return z_flat


def jacobian(decoder, z):
    z = z.squeeze()
    noutputs = decoder.data_dim
    z = z.repeat(noutputs, 1)
    z.requires_grad_(True)
    x = decoder(z)
    x.backward(torch.eye(noutputs))
    return z.grad.data


def reparameterize_riem(decoder, mu, logvar, n_samples=1, n_iterations=20):
    z_0 = reparametrize(mu, logvar)

    def decision_function(z):
        q_phi = torch.exp((z - mu) ** 2 / (2 * logvar.exp()))
        riem_measure = torch.det(jacobian(decoder, z))
        return q_phi * riem_measure
    z_t = z_0
    for t in range(n_iterations):
        generated_z = reparametrize(z_t, logvar)
        acceptance_ratio = decision_function(
            generated_z) / decision_function(z_t)
        u = torch.rand(1)
        if u < acceptance_ratio:
            z_t = generated_z
    return z_t


def sample_from_q(mu, logvar, n_samples=1):
    return reparametrize(mu, logvar, n_samples)


def sample_from_prior(latent_dim, n_samples=1):
    if CUDA:
        mu = torch.cuda.FloatTensor(n_samples, latent_dim).fill_(0)
        logvar = torch.cuda.FloatTensor(n_samples, latent_dim).fill_(0)
    else:
        mu = torch.zeros(n_samples, latent_dim)
        logvar = torch.zeros(n_samples, latent_dim)
    return reparametrize(mu, logvar)


class Encoder(nn.Module):
    def __init__(self, latent_dim, data_dim,
                 with_biasz=True, with_logvarz=True):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.with_logvarz = with_logvarz

        self.fc1 = nn.Linear(
            in_features=data_dim, out_features=latent_dim, bias=with_biasz)

        if with_logvarz:
            self.fc2 = nn.Linear(
                in_features=data_dim, out_features=latent_dim)

    def forward(self, x):
        """Forward pass of the encoder is encode."""
        muz = self.fc1(x)
        if self.with_logvarz:
            logvarz = self.fc2(x)
        else:
            logvarz = torch.zeros_like(muz)

        return muz, logvarz


class Decoder(nn.Module):
    def __init__(self, latent_dim, data_dim, n_layers=1,
                 nonlinearity=False,
                 with_biasx=True,
                 with_logvarx=True,
                 logvarx_true=None):
        """
        If with_logvarx is True, then the decoder predicts logvarx.
        Else, the decoder uses the cte logvarx_true as logvarx.
        """
        super(Decoder, self).__init__()

        if not with_logvarx:
            # logvarx is not predicted
            assert (logvarx_true is not None)
            # use the true value

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_layers = n_layers
        self.nonlinearity = nonlinearity
        self.with_logvarx = with_logvarx
        self.logvarx_true = logvarx_true

        # activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

        # layers
        self.layers = torch.nn.ModuleList()
        din = nn.Linear(
            in_features=latent_dim, out_features=2, bias=with_biasx)
        self.layers.append(din)

        for i in range(self.n_layers-2):
            dfc = nn.Linear(
                in_features=2, out_features=2, bias=with_biasx)
            self.layers.append(dfc)

        dfc = nn.Linear(
            in_features=2, out_features=data_dim, bias=with_biasx)
        self.layers.append(dfc)

        # layer for logvarx
        if with_logvarx:
            if self.n_layers == 1:
                dlogvarx = nn.Linear(
                    in_features=latent_dim, out_features=data_dim)
            else:
                dlogvarx = nn.Linear(
                    in_features=2, out_features=data_dim)
            self.layers.append(dlogvarx)

    def apply_nonlinearity(self, h):
        if self.nonlinearity is not None:
            if self.nonlinearity == 'relu':
                h = self.relu(h)
            elif self.nonlinearity == 'tanh':
                h = self.tanh(h)
            elif self.nonlinearity == 'softplus':
                h = self.softplus(h)
            elif self.nonlinearity == 'sigmoid':
                h = self.sigmoid(h)
        return h

    def forward(self, z):
        """Forward pass of the decoder is to decode."""
        if self.latent_dim == 1 and len(z.shape) == 1:
            z = z.unsqueeze(-1)
        h = self.layers[0](z)
        h = self.apply_nonlinearity(h)

        for i in range(1, self.n_layers-2):
            h = self.layers[i](h)
            h = self.apply_nonlinearity(h)

        if self.n_layers == 1:
            x = self.layers[0](z)
            if self.with_logvarx:
                logvarx = self.layers[1](z)
            else:
                logvarx = torch.zeros_like(x)
        else:
            x = self.layers[self.n_layers-1](h)
            if self.with_logvarx:
                logvarx = self.layers[self.n_layers](h)
            else:
                n_data, _ = x.shape
                logvarx = self.logvarx_true * torch.ones(
                    (n_data, 1)).to(DEVICE)
        return x, logvarx

    def generate(self, n_samples=1):
        """Generate from prior."""
        z = sample_from_prior(
            latent_dim=self.latent_dim, n_samples=n_samples)

        if n_samples == 1:
            z = z.unsqueeze(dim=0)
        else:
            z = z.unsqueeze(dim=1)

        x, logvarx = self.forward(z)
        return z, x, logvarx


class VAE(nn.Module):
    def __init__(self, latent_dim, data_dim, n_layers=1,
                 nonlinearity=False,
                 with_biasx=True, with_logvarx=True,
                 logvarx_true=None,
                 with_biasz=True, with_logvarz=True):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.encoder = Encoder(
            latent_dim=latent_dim,
            data_dim=data_dim,
            with_biasz=with_biasz,
            with_logvarz=with_logvarz)

        self.decoder = Decoder(
            latent_dim=latent_dim,
            data_dim=data_dim,
            n_layers=n_layers,
            nonlinearity=nonlinearity,
            with_biasx=with_biasx,
            with_logvarx=with_logvarx,
            logvarx_true=logvarx_true)

    def forward(self, x):
        muz, logvarz = self.encoder(x)
        z = reparametrize(muz, logvarz)
        res, logvarx = self.decoder(z)
        return res, logvarx, muz, logvarz


class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super(Discriminator, self).__init__()

        self.data_dim = data_dim

        # activation functions
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        n_layers = int(np.log2(self.data_dim)) + 1  # HACK - at least 1 layers
        n_layers = 10

        self.layers = torch.nn.ModuleList()

        for i in range(n_layers):
            layer = nn.Linear(
                in_features=data_dim,
                out_features=data_dim)
            self.layers.append(layer)

        # for i in range(n_layers):
        #    layer = nn.Linear(
        #        in_features=int(data_dim / (2 ** i)),
        #        out_features=int(data_dim / (2 ** (i+1))))
        #    self.layers.append(layer)

        last_layer = nn.Linear(
            in_features=self.layers[-1].out_features,
            out_features=1)
        self.layers.append(last_layer)

    def forward(self, x):
        """
        Forward pass of the discriminator is to take an image
        and output probability of the image being generated by the prior
        versus the learned approximation of the posterior.
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        prob = self.sigmoid(h)

        return prob
