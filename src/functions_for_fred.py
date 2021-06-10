"""VAE utils."""

import glob
import os

import nn_old
import numpy as np
import torch

import nn

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


def load_checkpoint(output, algo_name="vae", epoch_id=None):
    """load_checkpoint"""
    if epoch_id is None:
        ckpts = glob.glob("%s/train_%s/epoch_*_checkpoint.pth" % (output, algo_name))
        if len(ckpts) == 0:
            raise ValueError("No checkpoints found.")
        else:
            ckpts_ids_and_paths = [(int(f.split("_")[-2]), f) for f in ckpts]
            _, ckpt_path = max(ckpts_ids_and_paths, key=lambda item: item[0])
    else:
        # Load module corresponding to epoch_id
        ckpt_path = "%s/train_%s/epoch_%d_checkpoint.pth" % (
            output,
            algo_name,
            epoch_id,
        )
        print(ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise ValueError("No checkpoints found for epoch %d." % epoch_id)

    print("Found checkpoint. Getting: %s." % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    return ckpt


def load_module(output, module_name="encoder", epoch_id=None):
    ckpt = load_checkpoint(output=output, epoch_id=epoch_id)
    nn_architecture = ckpt["nn_architecture"]
    nn_type = nn_architecture["nn_type"]
    print("Loading %s from network of architecture: %s..." % (module_name, nn_type))
    assert nn_type in ["toy", "fc", "conv", "conv_plus", "conv_orig"]
    if nn_type == "toy":
        vae = toynn.VAE(
            latent_dim=nn_architecture["latent_dim"],
            data_dim=nn_architecture["data_dim"],
            n_layers=nn_architecture["n_decoder_layers"],
            nonlinearity=nn_architecture["nonlinearity"],
            with_biasx=nn_architecture["with_biasx"],
            with_logvarx=nn_architecture["with_logvarx"],
            logvarx_true=nn_architecture["logvarx_true"],
            with_biasz=nn_architecture["with_biasz"],
            with_logvarz=nn_architecture["with_logvarz"],
        )
        vae.to(DEVICE)
    elif nn_type == "fc":
        vae = nn.Vae(
            latent_dim=nn_architecture["latent_dim"],
            data_dim=nn_architecture["data_dim"],
            with_sigmoid=nn_architecture["with_sigmoid"],
            n_layers=nn_architecture["n_layers"],
            inner_dim=nn_architecture["inner_dim"],
            with_skip=nn_architecture["with_skip"],
            logvar=nn_architecture["logvar"],
        )
    elif nn_type == "conv":
        vae = nn.VaeConv(
            latent_dim=nn_architecture["latent_dim"],
            img_shape=nn_architecture["img_shape"],
            with_sigmoid=nn_architecture["with_sigmoid"],
        )
    elif nn_type == "conv_plus":
        vae = nn.VaeConvPlus(
            latent_dim=nn_architecture["latent_dim"],
            img_shape=nn_architecture["img_shape"],
            with_sigmoid=nn_architecture["with_sigmoid"],
        )
    else:
        img_shape = nn_architecture["img_shape"]
        latent_dim = nn_architecture["latent_dim"]
        with_sigmoid = nn_architecture["with_sigmoid"]
        n_blocks = nn_architecture["n_blocks"]
        vae = nn.VaeConvOrig(
            latent_dim=latent_dim,
            img_shape=img_shape,
            with_sigmoid=with_sigmoid,
            n_blocks=n_blocks,
        )
    modules = {}
    modules["encoder"] = vae.encoder
    modules["decoder"] = vae.decoder
    module = modules[module_name].to(DEVICE)
    module_ckpt = ckpt[module_name]
    module.load_state_dict(module_ckpt["module_state_dict"])
    return module


def latent_projection(output, dataset_path, epoch_id=None):
    dataset = np.load(dataset_path)
    encoder = load_module(output, module_name="encoder", epoch_id=epoch_id)
    dataset = torch.Tensor(dataset)
    dataset = torch.utils.data.TensorDataset(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    mus = []
    logvars = []
    for i, data in enumerate(loader):
        data = data[0].to(DEVICE)
        if len(data.shape) == 3:
            data = torch.unsqueeze(data, dim=0)

        assert len(data.shape) == 4
        mu, logvar = encoder(data)
        mus.append(np.array(mu.cpu().detach()))
        logvars.append(np.array(logvar.cpu().detach()))

    mus = np.array(mus).squeeze()
    logvars = np.array(logvars).squeeze()
    return mus, logvars


def load_module_old(output, algo_name="vae", module_name="encoder", epoch_id=None):
    print("Loading %s..." % module_name)
    ckpt = load_checkpoint(output=output, algo_name=algo_name, epoch_id=epoch_id)
    nn_architecture = ckpt["nn_architecture"]

    nn_type = nn_architecture["nn_type"]
    # print('{}'.format(nn_type))
    # fred: added conv_plus option below on October 20th, 2019
    assert nn_type in ["linear", "conv", "gan", "conv_plus"]

    if nn_type == "linear":
        vae = nn_old.Vae(
            latent_dim=nn_architecture["latent_dim"],
            data_dim=nn_architecture["data_dim"],
        )
    elif nn_type == "conv":
        vae = nn_old.VaeConv(
            latent_dim=nn_architecture["latent_dim"],
            img_shape=nn_architecture["img_shape"],
            spd=nn_architecture["spd"],
        )
    else:
        vae = nn_old.VaeGan(
            latent_dim=nn_architecture["latent_dim"],
            img_shape=nn_architecture["img_shape"],
        )
    vae.to(DEVICE)

    modules = {}
    modules["encoder"] = vae.encoder
    modules["decoder"] = vae.decoder
    module = modules[module_name]
    module_ckpt = ckpt[module_name]
    module.load_state_dict(module_ckpt["module_state_dict"])

    return module


def latent_projection_old(output, dataset_path, epoch_id=None):
    dataset = np.load(dataset_path)
    encoder = load_module_old(output, module_name="encoder", epoch_id=epoch_id)
    dataset = torch.Tensor(dataset)
    dataset = torch.utils.data.TensorDataset(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    mus = []
    for i, data in enumerate(loader):
        data = data[0].to(DEVICE)
        if len(data.shape) == 3:
            data = torch.unsqueeze(data, dim=0)

        assert len(data.shape) == 4
        mu, logvar = encoder(data)
        mus.append(np.array(mu.cpu().detach()))

    mus = np.array(mus).squeeze()
    return mus


##########


def reconstruction(
    output,
    z,
    # algo_name='vae',
    epoch_id=None,
):
    """reconstruction"""
    decoder = load_module(output, module_name="decoder", epoch_id=epoch_id)
    recon, _ = decoder(z)
    recon = recon.cpu().detach().numpy()
    return recon


def reconstruction_old(
    output,
    z,
    # algo_name='vae',
    epoch_id=None,
):
    """
    reconstruction_old
    """
    decoder = load_module_old(output, module_name="decoder", epoch_id=epoch_id)
    recon, _ = decoder(z)
    recon = recon.cpu().detach().numpy()
    return recon
