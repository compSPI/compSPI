"""Ray data processing pipeline."""

import functools
import logging
import numpy as np
import os
import random
import time

import ray

from hyperopt import hp
from ray import tune
from ray.tune import Trainable
from ray.tune.logger import CSVLogger, JsonLogger
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

import torch
import torch.autograd
from torch.nn import functional as F
import torch.optim
import torch.utils.data
import visdom

import datasets
import losses
import nn
import train_utils

import warnings
warnings.filterwarnings("ignore")

SERVER_NAME = 'slacgpu'

VISDOM = True if SERVER_NAME == 'gne' else False

DEBUG = False

DATASET_NAME = 'cryo_exp_3d'

# Hardware
CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

# Seed
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# NN architecture
IMG_SHAPE = (1, 128, 128)
DATA_DIM = functools.reduce((lambda x, y: x * y), IMG_SHAPE)
LATENT_DIM = 3
NN_TYPE = 'conv_orig'
SPD = False
assert NN_TYPE in ['toy', 'fc', 'conv', 'conv_plus', 'conv_orig']
assert NN_TYPE == 'fc' if SPD else True

NN_ARCHITECTURE = {
    'img_shape': IMG_SHAPE,
    'data_dim': DATA_DIM,
    'latent_dim': LATENT_DIM,
    'nn_type': NN_TYPE,
    'with_sigmoid': True,
    'spd': SPD}

# Train params

BATCH_SIZES = {15: 128, 25: 64, 64: 32, 90: 32, 96: 32, 100: 8, 128: 8}
BATCH_SIZE = BATCH_SIZES[IMG_SHAPE[1]]
FRAC_TEST = 0.1
FRAC_VAL = 0.2
N_SES_DEBUG = 3

AXIS = {'fmri': 3, 'mri': 1, 'seg': 1}

RECONSTRUCTIONS = ('bce_on_intensities', 'adversarial')
REGULARIZATIONS = ('kullbackleibler')
WEIGHTS_INIT = 'xavier'
REGU_FACTOR = 0.003

N_EPOCHS = 257

LR = 15e-6
if 'adversarial' in RECONSTRUCTIONS:
    LR = 0.001  # 0.002 # 0.0002

TRAIN_PARAMS = {
    'dataset_name': DATASET_NAME,
    'frac_val': FRAC_VAL,
    'lr': LR,
    'batch_size': BATCH_SIZE,
    'beta1': 0.9,
    'beta2': 0.999,
    'weights_init': WEIGHTS_INIT,
    'reconstructions': RECONSTRUCTIONS,
    'regularizations': REGULARIZATIONS,
    'lambda_regu': 1.,
    'lambda_adv': 1.
    }

if DEBUG:
    N_EPOCHS = 2
    N_FILEPATHS = 10
    FRAC_VAL = 0.5

PRINT_INTERVAL = 10
CKPT_PERIOD = 5


class Train(Trainable):

    def _setup(self, config):
        train_params = TRAIN_PARAMS
        train_params['lr'] = config.get('lr')
        train_params['lambda_regu'] = config.get(
                'lambda_regu')
        train_params['lambda_adv'] = config.get(
                'lambda_adv')
        train_params['dataset_name'] = config.get('dataset_name')
        train_params['class_2d'] = config.get('class_2d')

        nn_architecture = NN_ARCHITECTURE
        nn_architecture['latent_dim'] = config.get('latent_dim')
        nn_architecture['n_blocks'] = config.get('n_blocks')

        train_dataset, val_dataset = datasets.get_datasets(
                dataset_name=train_params['dataset_name'],
                frac_val=train_params['frac_val'],
                batch_size=train_params['batch_size'],
                img_shape=nn_architecture['img_shape'],
                class_2d=train_params['class_2d']
                )

        logging.info(
            'Train: %s' % train_utils.get_logging_shape(
                train_dataset))
        logging.info(
            'Val: %s' % train_utils.get_logging_shape(
                val_dataset))

        logging.info('NN architecture: ')
        logging.info(nn_architecture)
        logging.info('Training parameters:')
        logging.info(train_params)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_params['batch_size'],
            shuffle=True, **KWARGS)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=train_params['batch_size'],
            shuffle=True, **KWARGS)

        m, o, s, t, v = train_utils.init_training(
            self.logdir, nn_architecture, train_params)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.modules = modules
        self.optimizers = optimizers
        self.start_epoch = start_epoch

        self.train_losses_all_epochs = train_losses_all_epochs
        self.val_losses_all_epochs = val_losses_all_epochs

        self.train_params = train_params
        self.nn_architecture = nn_architecture

    def _train_iteration(self):
        """
        One epoch.
        - modules: a dict with the bricks of the model,
        eg. encoder, decoder, discriminator, depending on the architecture
        - optimizers: a dict with optimizers corresponding to each module.
        """
        start = time.time()

        epoch = self._iteration
        nn_architecture = self.nn_architecture
        train_params = self.train_params

        lambda_regu = train_params['lambda_regu']
        lambda_adv = train_params['lambda_adv']

        for module in self.modules.values():
            module.train()

        if VISDOM:
            train_vis = visdom.Visdom()
            train_vis.env = 'train_images'
            data_win = None
            recon_win = None
            from_prior_win = None

        total_loss_reconstruction = 0
        total_loss_regularization = 0
        if 'adversarial' in train_params['reconstructions']:
            total_loss_discriminator = 0
            total_loss_generator = 0
        total_loss = 0

        n_data = len(self.train_loader.dataset)
        n_batches = len(self.train_loader)

        for batch_idx, batch_data in enumerate(
                self.train_loader):
            if DEBUG and batch_idx < n_batches - 3:
                continue

            batch_data = batch_data.to(DEVICE)
            n_batch_data = len(batch_data)

            for optimizer in self.optimizers.values():
                optimizer.zero_grad()

            encoder = self.modules['encoder']
            decoder = self.modules['decoder']

            mu, logvar = encoder(batch_data)

            z = nn.sample_from_q(
                mu, logvar).to(DEVICE)
            batch_recon, scale_b = decoder(z)

            z_from_prior = nn.sample_from_prior(
                nn_architecture['latent_dim'],
                n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, scale_b_from_prior = decoder(
                z_from_prior)

            if 'adversarial' in train_params['reconstructions']:
                # From:
                # Autoencoding beyond pixels using a learned similarity metric
                # arXiv:1512.09300v2
                discriminator = self.modules['discriminator_reconstruction']
                real_labels = torch.full((n_batch_data, 1), 1, device=DEVICE)
                fake_labels = torch.full((n_batch_data, 1), 0, device=DEVICE)

                # -- Update DiscriminatorGan
                labels_data, h_data, _ = discriminator(
                    batch_data)
                labels_recon, h_recon, h_logvar_recon = discriminator(
                    batch_recon.detach())
                labels_from_prior, _, _ = discriminator(
                    batch_from_prior.detach())

                loss_dis_data = F.binary_cross_entropy(
                    labels_data,
                    real_labels)
                loss_dis_from_prior = F.binary_cross_entropy(
                    labels_from_prior,
                    fake_labels)

                # TODO(nina): add loss_dis_recon
                loss_discriminator = lambda_adv * (
                    loss_dis_data
                    + loss_dis_from_prior)

                # Fill gradients on discriminator only
                loss_discriminator.backward(retain_graph=True)

                # Need to do optimizer step here, as gradients
                # of the reconstruction with discriminator features
                # may fill the discriminator's weights and we do not
                # update the discriminator with the reconstruction loss.
                self.optimizers['discriminator_reconstruction'].step()

                # -- Update Generator/DecoderGAN
                # Note that we need to do a forward pass with detached vars
                # in order not to propagate gradients through the encoder
                batch_recon_detached, _ = decoder(z.detach())
                # Note that we don't need to do it for batch_from_prior
                # as it doesn't come from the encoder

                labels_recon, _, _ = discriminator(
                    batch_recon_detached)
                labels_from_prior, _, _ = discriminator(
                    batch_from_prior)

                loss_generator_recon = F.binary_cross_entropy(
                    labels_recon,
                    real_labels)

                # TODO(nina): add loss_generator_from_prior
                loss_generator = lambda_adv * loss_generator_recon

                # Fill gradients on generator only
                loss_generator.backward()

            if 'mse_on_intensities' in train_params['reconstructions']:
                loss_reconstruction = losses.mse_on_intensities(
                    batch_data, batch_recon, scale_b)

                # Fill gradients on encoder and generator
                loss_reconstruction.backward(retain_graph=True)

            if 'bce_on_intensities' in train_params['reconstructions']:
                loss_reconstruction = losses.bce_on_intensities(
                    batch_data, batch_recon, scale_b)

                # Fill gradients on encoder and generator
                loss_reconstruction.backward(retain_graph=True)

            if 'mse_on_features' in train_params['reconstructions']:
                # TODO(nina): Investigate stat interpretation
                # of using the logvar from the recon
                loss_reconstruction = losses.mse_on_features(
                    h_recon, h_data, h_logvar_recon)
                # Fill gradients on encoder and generator
                # but not on discriminator
                loss_reconstruction.backward(retain_graph=True)

            if 'kullbackleibler' in train_params['regularizations']:
                loss_regularization = lambda_regu * losses.kullback_leibler(
                    mu, logvar)
                # Fill gradients on encoder only
                loss_regularization.backward()

            if 'kullbackleibler_circle' in train_params['regularizations']:
                loss_regularization = losses.kullback_leibler_circle(
                        mu, logvar)
                # Fill gradients on encoder only
                loss_regularization.backward()

            if 'on_circle' in train_params['regularizations']:
                loss_regularization = losses.on_circle(
                        mu, logvar)
                # Fill gradients on encoder only
                loss_regularization.backward()

            self.optimizers['encoder'].step()
            self.optimizers['decoder'].step()

            loss = loss_reconstruction + loss_regularization
            if 'adversarial' in train_params['reconstructions']:
                loss += loss_discriminator + loss_generator

            if batch_idx % PRINT_INTERVAL == 0:
                # TODO(nina): Why didn't we need .mean() on 64x64?
                if 'adversarial' in train_params['reconstructions']:
                    self.print_train_logs(
                        epoch,
                        batch_idx, n_batches, n_data, n_batch_data,
                        loss, loss_reconstruction, loss_regularization,
                        loss_discriminator, loss_generator,
                        labels_data.mean(),
                        labels_recon.mean(),
                        labels_from_prior.mean())
                else:
                    self.print_train_logs(
                        epoch,
                        batch_idx, n_batches, n_data, n_batch_data,
                        loss, loss_reconstruction, loss_regularization)

                # Visdom first images of batch
                # TODO(nina): Why does it print black images for batch_data??
                # print(torch.sum(batch_data[0]))
                if VISDOM:
                    height = 150 * nn_architecture['img_shape'][1] / 64
                    width = 150 * nn_architecture['img_shape'][2] / 64
                    data_win = train_vis.image(
                        batch_data[0],
                        win=data_win,
                        opts=dict(
                            title='Train Epoch {}: Data'.format(epoch),
                            height=height, width=width))
                    recon_win = train_vis.image(
                        batch_recon[0],
                        win=recon_win,
                        opts=dict(
                            title='Train Epoch {}: Reconstruction'.format(
                                epoch),
                            height=height, width=width))
                    from_prior_win = train_vis.image(
                        batch_from_prior[0],
                        win=from_prior_win,
                        opts=dict(
                            title='Train Epoch {}: From prior'.format(epoch),
                            height=height, width=width))

            total_loss_reconstruction += loss_reconstruction.item()
            total_loss_regularization += loss_regularization.item()
            if 'adversarial' in train_params['reconstructions']:
                total_loss_discriminator += loss_discriminator.item()
                total_loss_generator += loss_generator.item()
            total_loss += loss.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        if 'adversarial' in train_params['reconstructions']:
            average_loss_discriminator = total_loss_discriminator / n_data
            average_loss_generator = total_loss_generator / n_data
        average_loss = total_loss / n_data

        logging.info('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, average_loss))

        end = time.time()

        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        if 'adversarial' in train_params['reconstructions']:
            train_losses['discriminator'] = average_loss_discriminator
            train_losses['generator'] = average_loss_generator
        train_losses['total'] = average_loss
        train_losses['otal_time'] = end - start

        self.train_losses_all_epochs.append(train_losses)

    def _train(self):
        self._train_iteration()
        return self._test()

    def _test(self):

        start = time.time()

        epoch = self._iteration
        nn_architecture = self.nn_architecture
        train_params = self.train_params

        for module in self.modules.values():
            module.eval()

        if VISDOM:
            vis = visdom.Visdom()
            vis.env = 'val_images'
            data_win = None
            recon_win = None
            from_prior_win = None

        total_loss_reconstruction = 0
        total_loss_regularization = 0
        if 'adversarial' in train_params['reconstructions']:
            total_loss_discriminator = 0
            total_loss_generator = 0
        total_loss = 0

        n_data = len(self.val_loader.dataset)
        n_batches = len(self.val_loader)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                if DEBUG and batch_idx < n_batches - 3:
                    continue
                batch_data = batch_data.to(DEVICE)
                n_batch_data = batch_data.shape[0]

                encoder = self.modules['encoder']
                decoder = self.modules['decoder']

                mu, logvar = encoder(batch_data)
                z = nn.sample_from_q(mu, logvar).to(DEVICE)
                batch_recon, scale_b = decoder(z)

                z_from_prior = nn.sample_from_prior(
                    nn_architecture['latent_dim'],
                    n_samples=n_batch_data).to(DEVICE)
                batch_from_prior, scale_b_from_prior = decoder(
                    z_from_prior)

                if 'adversarial' in train_params['reconstructions']:
                    discriminator = self.modules[
                        'discriminator_reconstruction']
                    real_labels = torch.full(
                        (n_batch_data, 1), 1, device=DEVICE)
                    fake_labels = torch.full(
                        (n_batch_data, 1), 0, device=DEVICE)

                    # -- Compute DiscriminatorGan Loss
                    labels_data, h_data, _ = discriminator(batch_data)
                    labels_recon, h_recon, h_logvar_recon = discriminator(
                        batch_recon.detach())
                    labels_from_prior, _, _ = discriminator(
                        batch_from_prior.detach())

                    loss_dis_data = F.binary_cross_entropy(
                        labels_data,
                        real_labels)
                    loss_dis_from_prior = F.binary_cross_entropy(
                        labels_from_prior,
                        fake_labels)

                    # TODO(nina): add loss_dis_recon
                    # loss_dis_recon = F.binary_cross_entropy(
                    #    labels_recon,
                    #    fake_labels)
                    loss_discriminator = (
                        loss_dis_data
                        + loss_dis_from_prior)

                    # -- Compute Generator/DecoderGAN Loss
                    # Note that we need to do a forward pass with detached vars
                    # in order not to propagate gradients through the encoder
                    batch_recon_detached, _ = decoder(z.detach())
                    # Note that we don't need to do it for
                    # batch_from_prior
                    # as it doesn't come from the encoder

                    labels_recon, _, _ = discriminator(
                        batch_recon_detached)
                    labels_from_prior, _, _ = discriminator(
                        batch_from_prior)

                    loss_generator_recon = F.binary_cross_entropy(
                        labels_recon,
                        real_labels)

                    # TODO(nina): add loss_generator_from_prior
                    loss_generator = loss_generator_recon

                if 'mse_on_intensities' in train_params['reconstructions']:
                    loss_reconstruction = losses.mse_on_intensities(
                        batch_data, batch_recon, scale_b)

                if 'bce_on_intensities' in train_params['reconstructions']:
                    loss_reconstruction = losses.bce_on_intensities(
                        batch_data, batch_recon, scale_b)

                if 'mse_on_features' in train_params['reconstructions']:
                    # TODO(nina): Investigate stat interpretation
                    # of using the logvar from the recon
                    loss_reconstruction = losses.mse_on_features(
                        h_recon, h_data, h_logvar_recon)

                if 'kullbackleibler' in train_params['regularizations']:
                    loss_regularization = losses.kullback_leibler(
                        mu, logvar)
                if 'kullbackleibler_circle' in train_params['regularizations']:
                    loss_regularization = losses.kullback_leibler_circle(
                            mu, logvar)

                if 'on_circle' in train_params['regularizations']:
                    loss_regularization = losses.on_circle(
                            mu, logvar)

                loss = loss_reconstruction + loss_regularization
                if 'adversarial' in train_params['reconstructions']:
                    loss += loss_discriminator + loss_generator

                total_loss_reconstruction += loss_reconstruction.item()
                total_loss_regularization += loss_regularization.item()
                if 'adversarial' in train_params['reconstructions']:
                    total_loss_discriminator += loss_discriminator.item()
                    total_loss_generator += loss_generator.item()
                total_loss += loss.item()

                if batch_idx == n_batches - 1:
                    # On last batch: visualize
                    batch_data = batch_data.cpu().numpy()
                    batch_recon = batch_recon.cpu().numpy()
                    batch_from_prior = batch_from_prior.cpu().numpy()

                    if VISDOM:
                        # Visdom first images of last batch
                        height = 150 * nn_architecture['img_shape'][1] / 64
                        width = 150 * nn_architecture['img_shape'][2] / 64
                        data_win = vis.image(
                            batch_data[0][0]+0.5,
                            win=data_win,
                            opts=dict(
                                title='Val Epoch {}: Data'.format(epoch),
                                height=height, width=width))
                        recon_win = vis.image(
                            batch_recon[0][0],
                            win=recon_win,
                            opts=dict(
                                title='Val Epoch {}: Reconstruction'.format(
                                    epoch),
                                height=height, width=width))
                        from_prior_win = vis.image(
                            batch_from_prior[0][0],
                            win=from_prior_win,
                            opts=dict(
                                title='Val Epoch {}: From prior'.format(epoch),
                                height=height, width=width))

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        if 'adversarial' in train_params['reconstructions']:
            average_loss_discriminator = total_loss_discriminator / n_data
            average_loss_generator = total_loss_generator / n_data
        average_loss = total_loss / n_data
        print('====> Val set loss: {:.4f}'.format(average_loss))

        end = time.time()

        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        if 'adversarial' in train_params['reconstructions']:
            val_losses['discriminator'] = average_loss_discriminator
            val_losses['generator'] = average_loss_generator
        val_losses['total'] = average_loss
        val_losses['total_time'] = end - start

        if np.isnan(average_loss) or np.isinf(average_loss):
            raise ValueError('Val loss is too large.')

        self.val_losses_all_epochs.append(val_losses)
        return {'average_loss': average_loss}

    def _save(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = self.logdir
        epoch = self._iteration
        train_utils.save_checkpoint(
            dir_path=checkpoint_dir,
            nn_architecture=self.nn_architecture,
            train_params=self.train_params,
            epoch=epoch,
            modules=self.modules,
            optimizers=self.optimizers,
            train_losses_all_epochs=self.train_losses_all_epochs,
            val_losses_all_epochs=self.val_losses_all_epochs)
        checkpoint_path = os.path.join(
            checkpoint_dir, 'epoch_%d_checkpoint.pth' % epoch)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        epoch_id = None  # HACK: restore last one
        train_dir = os.path.dirname(checkpoint_path)
        output = os.path.dirname(train_dir)
        for module_name, module in self.modules.items():
            self.modules[module_name] = train_utils.load_module_state(
                output=output,
                module_name=module_name,
                module=module,
                epoch_id=epoch_id)

    def print_train_logs(self,
                         epoch,
                         batch_idx, n_batches, n_data, n_batch_data,
                         loss,
                         loss_reconstruction, loss_regularization,
                         loss_discriminator=0, loss_generator=0,
                         dx=0, dgex=0, dgz=0):

        loss = loss / n_batch_data
        loss_reconstruction = loss_reconstruction / n_batch_data
        loss_regularization = loss_regularization / n_batch_data
        loss_discriminator = loss_discriminator / n_batch_data
        loss_generator = loss_generator / n_batch_data

        string_base = ('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}'
                       + '\nReconstruction: {:.6f}, Regularization: {:.6f}')

        if 'adversarial' in TRAIN_PARAMS['reconstructions']:
            string_base += (
                ', Discriminator: {:.6f}; Generator: {:.6f},'
                + 'D(x): {:.3f}, D(G(E(x))): {:.3f}, D(G(z)): {:.3f}')

        if 'adversarial' not in TRAIN_PARAMS['reconstructions']:
            logging.info(
                string_base.format(
                    epoch, batch_idx * n_batch_data, n_data,
                    100. * batch_idx / n_batches,
                    loss, loss_reconstruction, loss_regularization))
        else:
            logging.info(
                string_base.format(
                    epoch, batch_idx * n_batch_data, n_data,
                    100. * batch_idx / n_batches,
                    loss, loss_reconstruction, loss_regularization,
                    loss_discriminator, loss_generator,
                    dx, dgex, dgz))


def init():
    logging.getLogger().setLevel(logging.INFO)
    logging.info('start')


if __name__ == "__main__":
    init()

    ray.init()
    # ray.init(
    #     address=os.environ['ip_head'],
    #     redis_password=os.environ['redis_password'])

    search_space = {
        'dataset_name': 'cryo_exp_class_2d',
        'class_2d': 39,
        'lr': hp.loguniform(
            'lr',
            low=np.log(0.0001),
            high=np.log(0.01)),
        'latent_dim': hp.choice('latent_dim', [3, 4]),
        'n_blocks': hp.choice('n_blocks', [5, 6]),
        'lambda_regu': hp.loguniform(
            'lambda_regu',
            low=np.log(0.001), high=np.log(4.)),
        'lambda_adv': hp.loguniform(
            'lambda_adv',
            low=np.log(0.001), high=np.log(4.)),
    }

    hyperband_sched = AsyncHyperBandScheduler(
        time_attr='training_iteration',
        metric='average_loss',
        brackets=1,
        reduction_factor=8,
        mode='min')

    hyperopt_search = HyperOptSearch(
        search_space,
        metric='average_loss',
        mode='min',
        max_concurrent=257)

    analysis = tune.run(
        Train,
        local_dir='/results',
        name='output_cryo_exp_class_2d',
        scheduler=hyperband_sched,
        search_alg=hyperopt_search,
        loggers=[JsonLogger, CSVLogger],
        queue_trials=True,
        reuse_actors=True,
        **{
            'stop': {
                'training_iteration': N_EPOCHS,
            },
            'resources_per_trial': {
                'cpu': 4,
                'gpu': 1
            },
            'max_failures': 1,
            'num_samples': 257,
            'checkpoint_freq': CKPT_PERIOD,
            'checkpoint_at_end': True,
            'config': search_space})
