"""Utils to factorize code for both pipelines."""

import glob
import logging
import os

import numpy as np
import torch
import torch.nn as tnn

from geomstats.geometry.spd_matrices_space import SPDMatricesSpace

import nn
import toynn

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

CKPT_PERIOD = 1


W_INIT, B_INIT, NONLINEARITY_INIT = (
    {0: [[1.0], [0.0]],
     1: [[1.0, 0.0], [0.0, 1.0]]},
    {0: [0.0, 0.0],
     1: [0.01935, -0.02904]},
    'softplus')


def init_xavier_normal(m):
    if type(m) == tnn.Linear:
        tnn.init.xavier_normal_(m.weight)
    if type(m) == tnn.Conv2d:
        tnn.init.xavier_normal_(m.weight)


def init_kaiming_normal(m):
    if type(m) == tnn.Linear:
        tnn.init.kaiming_normal_(m.weight)
    if type(m) == tnn.Conv2d:
        tnn.init.kaiming_normal_(m.weight)


def init_custom(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_function(weights_init='xavier'):
    if weights_init == 'xavier':
        return init_xavier_normal
    elif weights_init == 'kaiming':
        return init_kaiming_normal
    elif weights_init == 'custom':
        return init_custom
    else:
        raise NotImplementedError(
            'This weight initialization is not implemented.')


def init_modules_and_optimizers(nn_architecture, train_params):
    modules = {}
    optimizers = {}

    nn_type = nn_architecture['nn_type']
    latent_dim = nn_architecture['latent_dim']
    data_dim = nn_architecture['data_dim']

    lr = train_params['lr']
    beta1 = train_params['beta1']
    beta2 = train_params['beta2']

    assert nn_type in ['toy', 'fc', 'conv', 'conv_plus', 'conv_orig']

    # Modules
    if nn_type == 'toy':
        vae = toynn.VAE(
            latent_dim=latent_dim,
            data_dim=data_dim,
            n_layers=nn_architecture['n_decoder_layers'],
            nonlinearity=nn_architecture['nonlinearity'],
            with_biasx=nn_architecture['with_biasx'],
            with_logvarx=nn_architecture['with_logvarx'],
            logvarx_true=nn_architecture['logvarx_true'],
            with_biasz=nn_architecture['with_biasz'],
            with_logvarz=nn_architecture['with_logvarz'])
        vae.to(DEVICE)

        logging.info('Values of VAE\'s decoder parameters before training:')
        decoder = vae.decoder
        for name, param in decoder.named_parameters():
            logging.info(name)
            logging.info(param.data)

    elif nn_type == 'fc':
        with_sigmoid = nn_architecture['with_sigmoid']
        n_layers = nn_architecture['n_layers']
        inner_dim = nn_architecture['inner_dim']
        with_skip = nn_architecture['with_skip']
        logvar = nn_architecture['logvar']
        vae = nn.Vae(
            latent_dim=latent_dim,
            data_dim=data_dim,
            with_sigmoid=with_sigmoid,
            n_layers=n_layers,
            inner_dim=inner_dim,
            with_skip=with_skip,
            logvar=logvar).to(DEVICE)
    elif nn_type == 'conv':
        img_shape = nn_architecture['img_shape']
        with_sigmoid = nn_architecture['with_sigmoid']
        vae = nn.VaeConv(
            latent_dim=latent_dim,
            img_shape=img_shape,
            with_sigmoid=with_sigmoid).to(DEVICE)
    elif nn_type == 'conv_plus':
        img_shape = nn_architecture['img_shape']
        with_sigmoid = nn_architecture['with_sigmoid']
        n_encoder_blocks = nn_architecture['n_encoder_blocks']
        n_decoder_blocks = nn_architecture['n_decoder_blocks']
        vae = nn.VaeConvPlus(
            latent_dim=latent_dim,
            img_shape=img_shape,
            with_sigmoid=with_sigmoid,
            n_encoder_blocks=n_encoder_blocks,
            n_decoder_blocks=n_decoder_blocks).to(DEVICE)
    else:
        img_shape = nn_architecture['img_shape']
        with_sigmoid = nn_architecture['with_sigmoid']
        n_blocks = nn_architecture['n_blocks']
        vae = nn.VaeConvOrig(
            latent_dim=latent_dim,
            img_shape=img_shape,
            with_sigmoid=with_sigmoid,
            n_blocks=n_blocks).to(DEVICE)

    modules['encoder'] = vae.encoder
    modules['decoder'] = vae.decoder

    if 'adversarial' in train_params['reconstructions']:
        discriminator = nn.Discriminator(
            latent_dim=nn_architecture['latent_dim'],
            img_shape=nn_architecture['img_shape']).to(DEVICE)
        modules['discriminator_reconstruction'] = discriminator

    if 'adversarial' in train_params['regularizations']:
        discriminator = nn.Discriminator(
            latent_dim=nn_architecture['latent_dim'],
            img_shape=nn_architecture['img_shape']).to(DEVICE)
        modules['discriminator_regularization'] = discriminator

    # Optimizers
    optimizers['encoder'] = torch.optim.Adam(
        modules['encoder'].parameters(), lr=lr, betas=(beta1, beta2))
    optimizers['decoder'] = torch.optim.Adam(
        modules['decoder'].parameters(), lr=lr, betas=(beta1, beta2))

    if 'adversarial' in train_params['reconstructions']:
        optimizers['discriminator_reconstruction'] = torch.optim.Adam(
            modules['discriminator_reconstruction'].parameters(),
            lr=train_params['lr'],
            betas=(train_params['beta1'], train_params['beta2']))

    if 'adversarial' in train_params['regularizations']:
        optimizers['discriminator_regularization'] = torch.optim.Adam(
            modules['discriminator_regularization'].parameters(),
            lr=train_params['lr'],
            betas=(train_params['beta1'], train_params['beta2']))

    return modules, optimizers


def init_training(train_dir, nn_architecture, train_params):
    """Initialization: Load ckpts or init."""
    start_epoch = 0
    train_losses_all_epochs = []
    val_losses_all_epochs = []

    modules, optimizers = init_modules_and_optimizers(
        nn_architecture, train_params)

    path_base = os.path.join(train_dir, 'epoch_*_checkpoint.pth')
    ckpts = glob.glob(path_base)
    if len(ckpts) == 0:
        weights_init = train_params['weights_init']
        logging.info(
            'No checkpoints found. Initializing with %s.' % weights_init)

        for module_name, module in modules.items():
            if nn_architecture['nn_type'] == 'toy':
                if module_name == 'decoder':
                    continue
            module.apply(init_function(weights_init))

    else:
        ckpts_ids_and_paths = [
            (int(f.split('_')[-2]), f) for f in ckpts]
        ckpt_id, ckpt_path = max(
            ckpts_ids_and_paths, key=lambda item: item[0])
        logging.info('Found checkpoints. Initializing with %s.' % ckpt_path)
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        ckpt = torch.load(ckpt_path, map_location=map_location)
        # ckpt = torch.load(ckpt_path, map_location=DEVICE)
        for module_name in modules.keys():
            module = modules[module_name]
            optimizer = optimizers[module_name]
            module_ckpt = ckpt[module_name]
            module.load_state_dict(module_ckpt['module_state_dict'])
            optimizer.load_state_dict(
                module_ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            train_losses_all_epochs = ckpt['train_losses']
            val_losses_all_epochs = ckpt['val_losses']

    return (modules, optimizers, start_epoch,
            train_losses_all_epochs, val_losses_all_epochs)


def save_checkpoint(epoch, modules, optimizers, dir_path,
                    train_losses_all_epochs, val_losses_all_epochs,
                    nn_architecture, train_params):
    checkpoint = {}
    for module_name in modules.keys():
        module = modules[module_name]
        optimizer = optimizers[module_name]
        checkpoint[module_name] = {
            'module_state_dict': module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}
        checkpoint['epoch'] = epoch
        checkpoint['train_losses'] = train_losses_all_epochs
        checkpoint['val_losses'] = val_losses_all_epochs
        checkpoint['nn_architecture'] = nn_architecture
        checkpoint['train_params'] = train_params

    checkpoint_path = os.path.join(
        dir_path, 'epoch_%d_checkpoint.pth' % epoch)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(output, epoch_id=None):
    if epoch_id is None:
        ckpts = glob.glob(
            '%s/checkpoint_*/epoch_*_checkpoint.pth' % output)
        if len(ckpts) == 0:
            raise ValueError('No checkpoints found.')
        else:
            ckpts_ids_and_paths = [(int(f.split('_')[-2]), f) for f in ckpts]
            ckpt_id, ckpt_path = max(
                ckpts_ids_and_paths, key=lambda item: item[0])
    else:
        # Load module corresponding to epoch_id
        ckpt_path = '%s/checkpoint_%d/epoch_%d_checkpoint.pth' % (
                output, epoch_id, epoch_id)
        if not os.path.isfile(ckpt_path):
            raise ValueError(
                'No checkpoints found for epoch %d in output %s.' % (
                    epoch_id, output))

    print('Found checkpoint. Getting: %s.' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    return ckpt


def load_module_state(output, module, module_name, epoch_id=None):
    ckpt = load_checkpoint(
        output=output, epoch_id=epoch_id)

    module_ckpt = ckpt[module_name]
    module.load_state_dict(module_ckpt['module_state_dict'])

    return module


def load_module(output, module_name='encoder', epoch_id=None):
    ckpt = load_checkpoint(
        output=output, epoch_id=epoch_id)
    nn_architecture = ckpt['nn_architecture']

    nn_type = nn_architecture['nn_type']
    print('Loading %s from network of architecture: %s...' % (
        module_name, nn_type))
    assert nn_type in ['toy', 'fc', 'conv', 'conv_plus', 'conv_orig']

    if nn_type == 'toy':
        vae = toynn.VAE(
            latent_dim=nn_architecture['latent_dim'],
            data_dim=nn_architecture['data_dim'],
            n_layers=nn_architecture['n_decoder_layers'],
            nonlinearity=nn_architecture['nonlinearity'],
            with_biasx=nn_architecture['with_biasx'],
            with_logvarx=nn_architecture['with_logvarx'],
            logvarx_true=nn_architecture['logvarx_true'],
            with_biasz=nn_architecture['with_biasz'],
            with_logvarz=nn_architecture['with_logvarz'])
        vae.to(DEVICE)

    elif nn_type == 'fc':
        vae = nn.Vae(
            latent_dim=nn_architecture['latent_dim'],
            data_dim=nn_architecture['data_dim'],
            with_sigmoid=nn_architecture['with_sigmoid'],
            n_layers=nn_architecture['n_layers'],
            inner_dim=nn_architecture['inner_dim'],
            with_skip=nn_architecture['with_skip'],
            logvar=nn_architecture['logvar'])
    elif nn_type == 'conv':
        vae = nn.VaeConv(
            latent_dim=nn_architecture['latent_dim'],
            img_shape=nn_architecture['img_shape'],
            with_sigmoid=nn_architecture['with_sigmoid'])
    elif nn_type == 'conv_plus':
        vae = nn.VaeConvPlus(
            latent_dim=nn_architecture['latent_dim'],
            img_shape=nn_architecture['img_shape'],
            with_sigmoid=nn_architecture['with_sigmoid'])
    else:
        img_shape = nn_architecture['img_shape']
        latent_dim = nn_architecture['latent_dim']
        with_sigmoid = nn_architecture['with_sigmoid']
        n_blocks = nn_architecture['n_blocks']
        vae = nn.VaeConvOrig(
            latent_dim=latent_dim,
            img_shape=img_shape,
            with_sigmoid=with_sigmoid,
            n_blocks=n_blocks)

    modules = {}
    modules['encoder'] = vae.encoder
    modules['decoder'] = vae.decoder
    module = modules[module_name].to(DEVICE)
    module_ckpt = ckpt[module_name]
    module.load_state_dict(module_ckpt['module_state_dict'])

    return module


def get_logging_shape(tensor):
    shape = tensor.shape
    logging_shape = '(' + ('%s, ' * len(shape) % tuple(shape))[:-2] + ')'
    return logging_shape


def spd_feature_from_matrix(dataset, spd_feature='matrix'):
    """Transform everything in the np dataset"""
    assert type(dataset) == np.ndarray
    if spd_feature == 'matrix' or spd_feature == 'point':
        dataset = dataset
        assert dataset.ndim == 4
    else:
        _, _, n, _ = dataset.shape
        mat_identity = np.eye(n)
        spd_space = SPDMatricesSpace(n=n)
        dataset = dataset[:, 0, :, :]  # channels

        if spd_feature == 'vector':
            dataset = spd_space.vector_from_symmetric_matrix(dataset)
            dataset = np.expand_dims(dataset, axis=1)
        if spd_feature == 'log_matrix':
            dataset = spd_space.metric.log(
                base_point=mat_identity, point=dataset)
            dataset = np.expand_dims(dataset, axis=1)
        if spd_feature == 'log_vector':
            dataset = spd_space.metric.log(
                base_point=mat_identity, point=dataset)
            dataset = spd_space.vector_from_symmetric_matrix(dataset)
            dataset = np.expand_dims(dataset, axis=1)
    return dataset


def matrix_from_spd_feature(dataset, spd_feature='matrix'):
    if spd_feature == 'matrix' or spd_feature == 'point':
        return dataset
    _, vec_dim = dataset.shape
    n = int((np.sqrt(8 * vec_dim + 1) - 1) / 2)
    spd_space = SPDMatricesSpace(n=n)

    if spd_feature == 'vector':
        dataset = spd_space.symmetric_matrix_from_vector(dataset)
        return dataset
    if spd_feature == 'log_vector':
        dataset = spd_space.symmetric_matrix_from_vector(dataset)
        mat_identity = np.eye(n)
        dataset = spd_space.metric.exp(
            base_point=mat_identity, tangent_vec=dataset)
        return dataset
