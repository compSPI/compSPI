"""Contains classes of differentiables models for 3D reconstruction."""

import torch
import torch.nn as nn


class EncoderCryoAI(torch.nn.Module):
    """
    Encoder that predicts poses (rotations and translation) from images.

    Reference: Levy, Axel, et al. "CryoAI: Amortized Inference of Poses for
    Ab Initio Reconstruction of 3D Molecular Volumes from Real Cryo-EM
    Images." arXiv preprint arXiv:2203.08138 (2022).
    """

    def __init__(self, size, dim_rot=6, dim_trans=2, dim_features=256,
                 dim_hidden_features=256, dim_hidden_rot=128,
                 dim_hidden_trans=128, hidden_layers_features=2,
                 hidden_layers_rot=3, hidden_layers_trans=2,
                 n_filters=64, mask=None, coord_conv=True):
        """
        Initialization of the encoder.

        Parameters
        ----------
        size: int
            resolution of input images
        dim_rot: int
            Number of dimensions to represent rotations
        dim_trans: int
            Number of dimensions to represent translations
        dim_features: int
            Number of dimensions of high level features
        dim_hidden_features: int
            Number of hidden dimensions in the MLP
        dim_hidden_rot: int
            Number of hidden dimensions in the MLP
        dim_hidden_trans: int
            Number of hidden dimensions in the MLP
        hidden_layers_features: int
            Number of hidden layers in the MLP
        hidden_layers_rot: int
            Number of hidden layers in the MLP
        hidden_layers_trans: int
            Number of hidden layers in the MLP
        n_filters: int
            Number of filters in the first convolutional layer
        coord_conv: bool
            Use CoordConv layer first (Liu, Rosanne, et al. "An intriguing
            failing of convolutional neural networks and the coordconv
            solution." Advances in neural information processing systems 31
            (2018))
        mask: torch.Tensor
            Boolean tensor of size (D, D)
        """
        super(EncoderCryoAI, self).__init__()

        if mask is not None:
            self.apply_mask = True
            self.mask = mask[None, None, :, :]
        else:
            self.apply_mask = False

        self.coord_conv = coord_conv
        if coord_conv:
            self.coord_conv_layer = AddCoordinates(with_r=True)
            in_dim_cnn = 4
        else:
            in_dim_cnn = 1

        assert size >= 32

        cnn = [nn.Conv2d(in_dim_cnn, n_filters, 5, stride=1, padding=2)]

        cnn.append(nn.ReLU(inplace=True))
        cnn.append(nn.MaxPool2d(kernel_size=2))

        cnn.append(nn.Conv2d(n_filters, n_filters, 5, stride=1, padding=2))
        cnn.append(nn.ReLU(inplace=True))
        cnn.append(nn.BatchNorm2d(n_filters))
        cnn.append(nn.MaxPool2d(kernel_size=2))

        cnn.append(nn.Conv2d(n_filters, n_filters * 2, 3, stride=1, padding=1))
        cnn.append(nn.ReLU(inplace=True))
        cnn.append(
            nn.Conv2d(n_filters * 2, n_filters * 2, 3, stride=1, padding=1))
        cnn.append(nn.ReLU(inplace=True))
        cnn.append(nn.BatchNorm2d(n_filters * 2))
        cnn.append(nn.MaxPool2d(kernel_size=2))

        cnn.append(
            nn.Conv2d(n_filters * 2, n_filters * 4, 3, stride=1, padding=1))
        cnn.append(nn.ReLU(inplace=True))
        cnn.append(
            nn.Conv2d(n_filters * 4, n_filters * 4, 3, stride=1, padding=1))
        cnn.append(nn.ReLU(inplace=True))
        cnn.append(nn.BatchNorm2d(n_filters * 4))
        cnn.append(nn.MaxPool2d(kernel_size=2))

        cnn.append(
            nn.Conv2d(n_filters * 4, n_filters * 4, 3, stride=1, padding=1))
        cnn.append(nn.ReLU(inplace=True))
        cnn.append(nn.BatchNorm2d(n_filters * 4))
        cnn.append(nn.MaxPool2d(kernel_size=2))

        # state size (ndf*4) x D/32 x D/32 (64->1024, 128->4096)
        self.cnn = nn.Sequential(*cnn)

        self.out_dim_cnn = 4 * n_filters * (size // 32) * (size // 32)
        self.mlp = ResidLinearMLP(self.out_dim_cnn, hidden_layers_features,
                                  dim_hidden_features, dim_features,
                                  init='normal')

        self.rot_encoder = ResidLinearMLP(dim_features, hidden_layers_rot,
                                          dim_hidden_rot, dim_rot,
                                          init='normal')
        self.trans_encoder = ResidLinearMLP(dim_features, hidden_layers_trans,
                                            dim_hidden_trans, dim_trans,
                                            init='normal')

    def extract_features(self, img):
        """
        Extract high level visual features from images.

        Parameters
        ----------
        img: torch.Tensor
            Tensor of size (batch, 1, D, D)

        Returns
        -------
        features: torch.Tensor
            Tensor of size (batch, dim_features)
        """
        if self.coord_conv:
            x = self.coord_conv_layer(img)
        x = self.cnn(x)
        x = x.reshape(-1, self.out_dim_cnn)
        features = self.mlp(x)
        return features

    def extract_rot(self, features):
        """
        Extract rotations from high level visual features.

        Parameters
        ----------
        features: torch.Tensor
            Tensor of size (batch, dim_features)

        Returns
        -------
        rot: torch.Tensor
            Tensor of size (batch, dim_rot)
        """
        rot = self.rot_encoder(features)
        return rot

    def extract_trans(self, features):
        """
        Extract rotations from high level visual features.

        Parameters
        ----------
        features: torch.Tensor
            Tensor of size (batch, dim_features)

        Returns
        -------
        trans: torch.Tensor
            Tensor of size (batch, dim_trans)
        """
        trans = self.trans_encoder(features)
        return trans

    def forward(self, img):
        """
        Extract rotation and translation parameters from images.

        Parameters
        ----------
        img: torch.Tensor
            Tensor of size (batch, 1, D, D)

        Returns
        -------
        rot: torch.Tensor
            Tensor of size (batch, dim_rot)
        trans: torch.Tensor
            Tensor of size (batch, dim_trans)
        """
        if self.apply_mask:
            img = img * self.mask
        features = self.extract_features(img)
        rot = self.extract_rot(features)
        trans = self.extract_trans(features)
        return rot, trans


class ResidLinearMLP(nn.Module):
    """
    Residual MLP imported from cryoDRGN2.

    Reference: Zhong, Ellen D., et al. "CryoDRGN2: Ab initio neural
    reconstruction of 3D protein structures from real cryo-EM images."
    Proceedings of the IEEE/CVF International Conference on Computer Vision.
    2021
    """

    def __init__(self, in_dim, nlayers, hidden_dim, out_dim,
                 activation='relu', batchnorm=True, init=None):
        """
        Initialization of residual MLP.

        Parameters
        ----------
        in_dim: int
            number of dimensions of the input
        nlayers: int
            number of hidden layers
        hidden_dim: int
            number of hidden dimensions
        out_dim: int
            number of dimensions of the output
        activation: str
            activation function ('relu')
        batchnorm: bool
            use batch normalization
        init: str
            type of initialization ('normal', None)
        """
        super(ResidLinearMLP, self).__init__()

        if activation == 'relu':
            nl = nn.ReLU()
        else:
            raise NotImplementedError

        layers = [ResidLinear(in_dim,
                              init=init) if in_dim == hidden_dim else nn.Linear(
            in_dim, hidden_dim),
                  nl]

        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(nlayers - 1):
            layers.append(ResidLinear(hidden_dim, init=init))
            layers.append(nl)
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(
            ResidLinear(hidden_dim) if out_dim == hidden_dim else nn.Linear(
                hidden_dim, out_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Tensor of size (batch, in_dim)

        Returns
        -------
        out: torch.Tensor
            Tensor of size (batch, out_dim)
        """
        out = self.main(x)
        return out


class ResidLinear(nn.Module):
    """Residual linear layer."""

    def __init__(self, dim, init=None):
        """
        Initialization of residual linear layer.

        Parameters
        ----------
        dim: int
            number of dimensions of the input and output
        init: str
            type of initialization ('normal', None)
        """
        super(ResidLinear, self).__init__()
        self.linear = nn.Linear(dim, dim)
        if init == 'normal':
            init_weights_normal(self.linear)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Tensor of size (batch, dim)

        Returns
        -------
        out: torch.Tensor
            Tensor of size (batch, dim)
        """
        out = self.linear(x) + x
        return out


class AddCoordinates(nn.Module):
    """
    CoordConv layer.

    Reference :Liu, Rosanne, et al. "An intriguing failing of convolutional
    neural networks and the coordconv solution." Advances in neural
    information processing systems 31 (2018)
    """

    def __init__(self, with_r=False):
        """
        Initialization of CoordConv.

        Parameters
        ----------
        with_r: bool
            Add r coordinates
        """
        super(AddCoordinates, self).__init__()
        self.with_r = with_r

    def forward(self, img):
        """
        Forward pass.

        Parameters
        ----------
        img: torch.Tensor
            Tensor of size (batch, 1, D, D)

        Returns
        -------
        out: torch.Tensor
            Tensor of size (batch, 3/4, D, D)
        """
        batch_size, _, image_height, image_width = img.size()

        y_coords = 2.0 * torch.arange(image_height,
                                      device=img.device).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width,
                                      device=img.device).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0

        coords = torch.stack((y_coords, x_coords), dim=0)

        if self.with_r:
            rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
            rs = rs / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)

        out = torch.cat((coords.to(img.device), img), dim=1)

        return out


def init_weights_normal(m):
    """
    Initialization of a linear layer with kaiming normal distribution.

    Parameters
    ----------
    m: torch.nn.Module
        Linear torch module
    """
    if type(m) is nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu',
                                    mode='fan_out')
        if hasattr(m, 'bias'):
            nn.init.uniform_(m.bias, -1, 1)
