"""Utils for getting datasets."""

import csv
import glob
import gzip
import h5py
import importlib
import logging
import os
import pandas as pd
import pickle

import numpy as np
import torch
import torch.utils


import geomstats

from geomstats.geometry.spd_matrices_space import SPDMatricesSpace
from skimage import transform
from torchvision import datasets, transforms
from urllib import request

import toynn

CUDA = torch.cuda.is_available()
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

CRYO_DIR = '/data/cryo'
# CRYO_DIR = '/afs/slac.stanford.edu/u/bd/nmiolane/gpfs_home/data/cryo'
# CRYO_H5 = CRYO_DIR
CRYO_H5 = os.path.join(
    '/gpfs/slac/cryo/fs1/g/ML/vaegan/datasets',
    'exp/20181005-rib-TEM4/Sort')
NEURO_DIR = '/data/neuro'

NEURO_TRAIN_VAL_DIR = os.path.join(NEURO_DIR, 'train_val_datasets')
CRYO_TRAIN_VAL_DIR = os.path.join(CRYO_DIR, 'train_val_datasets')

N_NODES = 28
CORR_THRESH = 0.1
GAMMA = 1.0
N_GRAPHS = 86
ID_COEF = 4  # Make Positive definite

FRAC_VAL = 0.05

# TODO(nina): Reorganize:
# get_datasets provide train/val in np.array,
# get_loaders shuflles and transforms in tensors/loaders


def get_datasets(dataset_name, frac_val=FRAC_VAL, batch_size=8,
                 img_shape=None,
                 nn_architecture=None,
                 train_params=None,
                 synthetic_params=None,
                 class_2d=None,
                 kwargs=KWARGS):

    img_shape_no_channel = None
    if img_shape is not None:
        img_shape_no_channel = img_shape[1:]
    # TODO(nina): Consistency in datasets: add channels for all
    logging.info('Loading data from dataset: %s' % dataset_name)
    if dataset_name == 'mnist':
        train_dataset, val_dataset = get_dataset_mnist()
    elif dataset_name == 'omniglot':
        if img_shape_no_channel is not None:
            transform = transforms.Compose([
                transforms.Resize(img_shape_no_channel),
                transforms.ToTensor()])
        else:
            transform = transforms.ToTensor()
        dataset = datasets.Omniglot(
            '../data', download=True, transform=transform)
        train_dataset, val_dataset = split_dataset(
            dataset, frac_val=frac_val)
    elif dataset_name in [
            'cryo_sim',
            'randomrot1D_nodisorder',
            'randomrot1D_multiPDB',
            'randomrot_nodisorder']:
        dataset = get_dataset_cryo(dataset_name, img_shape_no_channel, kwargs)
        train_dataset, val_dataset = split_dataset(dataset)
    elif dataset_name == 'cryo_sphere':
        dataset = get_dataset_cryo_sphere(img_shape_no_channel, kwargs)
        train_dataset, val_dataset = split_dataset(dataset)
    elif dataset_name == 'cryo_exp':
        dataset = get_dataset_cryo_exp(img_shape_no_channel, kwargs)
        train_dataset, val_dataset = split_dataset(dataset)
    elif dataset_name == 'cryo_exp_class_2d':
        dataset = get_dataset_cryo_exp_class_2d(
            img_shape_no_channel, class_2d, kwargs)
        train_dataset, val_dataset = split_dataset(dataset)
    elif dataset_name == 'cryo_exp_3d':
        dataset = get_dataset_cryo_exp_3d(img_shape_no_channel, kwargs)
        train_dataset, val_dataset = split_dataset(dataset)
    elif dataset_name == 'connectomes':
        train_dataset, val_dataset = get_dataset_connectomes(
            img_shape_no_channel=img_shape_no_channel)
    elif dataset_name == 'connectomes_simu':
        train_dataset, val_dataset = get_dataset_connectomes_simu(
            img_shape_no_channel=img_shape_no_channel)
    elif dataset_name == 'connectomes_schizophrenia':
        train_dataset, val_dataset, _ = get_dataset_connectomes_schizophrenia()
    elif dataset_name in ['mri', 'segmentation', 'fmri']:
        train_loader, val_loader = get_loaders_brain(
            dataset_name, frac_val, batch_size, img_shape_no_channel, kwargs)
        return train_loader, val_loader
    elif dataset_name == 'synthetic':
        dataset = make_synthetic_dataset_and_decoder(
            synthetic_params=synthetic_params,
            nn_architecture=nn_architecture,
            train_params=train_params)
        train_dataset, val_dataset = split_dataset(dataset)
    else:
        raise ValueError('Unknown dataset name: %s' % dataset_name)

    return train_dataset, val_dataset


def split_dataset(dataset, frac_val=FRAC_VAL,
                  labels_path=None, save=False, data_dir=None):
    length = len(dataset)
    train_length = int((1 - frac_val) * length)
    train_dataset = dataset[:train_length]
    val_dataset = dataset[train_length:]
    return train_dataset, val_dataset


def get_shape_string(img_shape_no_channel):
    if len(img_shape_no_channel) == 2:
        shape_str = '%dx%d' % img_shape_no_channel
    elif len(img_shape_no_channel) == 3:
        shape_str = '%dx%dx%d' % img_shape_no_channel
    else:
        raise ValueError('Weird image shape.')
    return shape_str


def normalization_linear(dataset):
    for i in range(len(dataset)):
        data = dataset[i]
        min_data = np.min(data)
        max_data = np.max(data)
        dataset[i] = (data - min_data) / (max_data - min_data)
    return dataset


def add_channels(dataset, img_dim=2):
    if dataset.ndim == 3:
        dataset = np.expand_dims(dataset, axis=1)
    return dataset


def is_pos_def(x):
    eig, _ = np.linalg.eig(x)
    return (eig > 0).all()


def is_sym(x):
    return np.all(np.isclose(x, np.transpose(x), rtol=1e-4))


def is_spd(x):
    """Assumes the matrix is symmetric"""
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    elif x.ndim == 4:
        x = x[:, 0, :, :]
    _, n, _ = x.shape
    all_spd = True
    for i, one_mat in enumerate(x):
        if not is_pos_def(one_mat):
            print('problem pos def at %d' % i)
            print(np.linalg.eig(one_mat)[0])
        if not is_sym(one_mat):
            print('problem sym at %d' % i)
            print(one_mat - np.transpose(one_mat))

        all_spd = all_spd & is_sym(one_mat) & is_pos_def(one_mat)
    return all_spd


def r_pearson_from_z_score(mat):
    """Inverse Fisher transformation"""
    r_mat = np.tanh(mat)
    return r_mat


def get_dataset_mnist(img_shape_no_channel=(28, 28)):
    shape_str = get_shape_string(img_shape_no_channel)
    train_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'train_mnist_%s.npy' % shape_str)
    val_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'val_mnist_%s.npy' % shape_str)

    train_exists = os.path.isfile(train_path)
    val_exists = os.path.isfile(val_path)
    if train_exists and val_exists:
        print('Loading %s...' % train_path)
        print('Loading %s...' % val_path)
        train_dataset = np.load(train_path)
        val_dataset = np.load(val_path)
    else:
        filename = [
            ['training_images', 'train-images-idx3-ubyte.gz'],
            ['test_images', 't10k-images-idx3-ubyte.gz'],
            ['training_labels', 'train-labels-idx1-ubyte.gz'],
            ['test_labels', 't10k-labels-idx1-ubyte.gz']
            ]
        base_url = 'http://yann.lecun.com/exdb/mnist/'
        save_folder = '/neuro/train_val_datasets/'
        for name in filename:
            print('Downloading ' + name[1] + '...')
            request.urlretrieve(base_url + name[1], save_folder + name[1])
        print('Download complete.')

        mnist = {}

        for name in filename[:2]:
            with gzip.open(name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(
                    f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        for name in filename[-2:]:
            with gzip.open(name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open('mnist.pkl', 'wb') as f:
            pickle.dump(mnist, f)
        print('Save complete.')

        with open('mnist.pkl', 'rb') as f:
            mnist = pickle.load(f)  # training_labels, test_labels also

        dataset = mnist['training_images']
        dataset = add_channels(dataset, img_dim=2)
        dataset = dataset / 255  # normalization
        train_dataset, val_dataset = split_dataset(
                dataset, frac_val=FRAC_VAL)
        print('Saving %s...' % train_path)
        print('Saving %s...' % val_path)
        np.save(train_path, train_dataset)
        np.save(val_path, val_dataset)

    return train_dataset, val_dataset


def get_dataset_connectomes(img_shape_no_channel=(100, 100)):
    """
    Connectomes from HCP 1200:
    https://www.humanconnectome.org/storage/app/media/
    documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf
    """
    shape_str = get_shape_string(img_shape_no_channel)
    train_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'train_conn_%s.npy' % shape_str)
    val_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'val_conn_%s.npy' % shape_str)

    hcp_labels_path = os.path.join(
        NEURO_TRAIN_VAL_DIR,
        'hcp_labels.csv')

    train_exists = os.path.isfile(train_path)
    val_exists = os.path.isfile(val_path)
    if train_exists and val_exists:
        print('Loading %s...' % train_path)
        print('Loading %s...' % val_path)
        train_dataset = np.load(train_path)
        val_dataset = np.load(val_path)

    else:
        hcp_meta = pd.read_csv(
            '/neuro/HCP_PTN1200_recon2/hcp_metadata.csv')

        n_nodes = img_shape_no_channel[0]
        hcp_dir = os.path.join(
            NEURO_DIR, 'HCP_PTN1200_recon2')
        ts_dir = os.path.join(
                hcp_dir, 'node_timeseries/3T_HCP1200_MSMAll_d%d_ts2' % n_nodes)
        string_base = '%s/*.txt' % ts_dir

        all_paths = glob.glob(string_base)
        print('Found %d paths.' % len(all_paths))

        all_ts = []
        all_subject_ids = []
        all_genders = []
        all_ages = []
        for i, path in enumerate(all_paths):
            print('Extracting time series %s...' % path)
            ts = np.loadtxt(path)
            all_ts.append(ts)
            basename = os.path.basename(path)
            subject_id = basename.split('.')[0]
            print('subject id: %s' % subject_id)
            subject_id = int(subject_id)
            all_subject_ids.append(subject_id)

            hcp_meta_row = hcp_meta[hcp_meta['Subject'] == subject_id]

            gender = hcp_meta_row['Gender'].values[0]
            print(gender)
            age = hcp_meta_row['Age'].values[0]
            print(age)
            all_genders.append(gender)
            all_ages.append(age)
        all_ts = np.array(all_ts)

        n_data = all_ts.shape[0]
        print('Found %d data.' % n_data)
        all_connectomes = np.zeros((n_data, n_nodes, n_nodes))

        for i_ts in range(n_data):
            for i_node in range(n_nodes):
                for j_node in np.arange(i_node,  n_nodes, 1):
                    ts_of_node_i = all_ts[i_ts, :, i_node]
                    ts_of_node_j = all_ts[i_ts, :, j_node]
                    x = np.vstack([ts_of_node_i, ts_of_node_j])
                    corr = np.corrcoef(x)[0, 1]
                    all_connectomes[i_ts, i_node, j_node] = corr
                    all_connectomes[i_ts, j_node, i_node] = corr

        dataset = all_connectomes
        dataset = add_channels(dataset, img_dim=2)
        assert len(dataset.shape) == 4
        print('Dataset of shape:')
        print(dataset.shape)
        train_dataset, val_dataset = split_dataset(dataset)
        print('Saving %s...' % train_path)
        print('Saving %s...' % val_path)
        np.save(train_path, train_dataset)
        np.save(val_path, val_dataset)

        with open(hcp_labels_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Subject', 'Gender', 'Age'])
            for subject_id, gender, age in zip(
                    all_subject_ids, all_genders, all_ages):
                writer.writerow([subject_id, gender, age])

    return train_dataset, val_dataset


def get_dataset_connectomes_from_netmats(img_shape_no_channel=(100, 100),
                                         partial_corr=True):
    """
    Connectomes from HCP 1200:
    https://www.humanconnectome.org/storage/app/media/
    documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf
    """
    netmat_type = int(partial_corr) + 1

    shape_str = get_shape_string(img_shape_no_channel)
    train_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'train_conn_%s.npy' % shape_str)
    val_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'val_conn_%s.npy' % shape_str)

    train_exists = os.path.isfile(train_path)
    val_exists = os.path.isfile(val_path)
    if train_exists and val_exists:
        print('Loading %s...' % train_path)
        print('Loading %s...' % val_path)
        train_dataset = np.load(train_path)
        val_dataset = np.load(val_path)

    else:
        n_nodes = img_shape_no_channel[0]
        hcp_dir = os.path.join(
            NEURO_DIR, 'HCP_PTN1200_recon2')
        netmats_path = os.path.join(
            hcp_dir, 'netmats/3T_HCP1200_MSMAll_d%d_ts2/netmats%d.txt' % (
                n_nodes, netmat_type))
        print('Loading %s...' % netmats_path)
        netmats = np.loadtxt(netmats_path)
        netmats = netmats.reshape(-1, n_nodes, n_nodes)
        n_mats, _, _ = netmats.shape

        r_mats = r_pearson_from_z_score(netmats)

        # HACK
        r_mats = 1 / 4 * (r_mats + ID_COEF * np.tile(
            np.eye(n_nodes, n_nodes), (n_mats, 1, 1)))
        r_mats = np.abs(r_mats)

        r_mats = add_channels(r_mats, img_dim=2)

        dataset = r_mats

        assert len(dataset.shape) == 4

        # dataset = normalization_linear(dataset)

        train_dataset, val_dataset = split_dataset(dataset)
        print('Saving %s...' % train_path)
        print('Saving %s...' % val_path)
        np.save(train_path, train_dataset)
        np.save(val_path, val_dataset)

    return train_dataset, val_dataset


def get_dataset_connectomes_simu(img_shape_no_channel=(15, 15)):
    """
    Simulating a geodesic triangle.
    """
    shape_str = get_shape_string(img_shape_no_channel)
    train_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'train_conn_simu_%s.npy' % shape_str)
    val_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'val_conn_simu_%s.npy' % shape_str)

    train_exists = os.path.isfile(train_path)
    val_exists = os.path.isfile(val_path)
    if train_exists and val_exists:
        print('Loading %s...' % train_path)
        print('Loading %s...' % val_path)
        train_dataset = np.load(train_path)
        val_dataset = np.load(val_path)

    else:
        n, _ = img_shape_no_channel
        os.environ['GEOMSTATS_BACKEND'] = 'numpy'
        spd_space = SPDMatricesSpace(n=n)
        vec_dim = int(n * (n + 1) / 2)
        vec_a = np.zeros(vec_dim)
        vec_b = np.zeros(vec_dim)
        vec_c = np.zeros(vec_dim)

        cos_angle = np.cos(np.pi / 3)
        sin_angle = np.cos(np.pi / 3)
        vec_a[0] = cos_angle
        vec_a[1] = sin_angle
        vec_b[0] = -cos_angle
        vec_b[1] = sin_angle
        vec_c[0] = 0.
        vec_c[1] = -1.

        # mat_identity = np.eye(n)
        # sym_mat_a = spd_space.symmetric_matrix_from_vector(vec_a)
        # spd_mat_a = spd_space.metric.exp(
        #     base_point=mat_identity, tangent_vec=sym_mat_a)
        # sym_mat_b = spd_space.symmetric_matrix_from_vector(vec_b)
        # spd_mat_b = spd_space.metric.exp(
        #     base_point=mat_identity, tangent_vec=sym_mat_b)
        # sym_mat_c = spd_space.symmetric_matrix_from_vector(vec_c)
        # spd_mat_c = spd_space.metric.exp(
        #     base_point=mat_identity, tangent_vec=sym_mat_c)

        spd_mat_a = np.eye(n)  # spd_space.random_uniform()
        spd_mat_b = spd_space.random_uniform()
        spd_mat_c = spd_space.random_uniform()
        assert is_spd(spd_mat_a)
        assert is_spd(spd_mat_b)
        assert is_spd(spd_mat_c)

        vec_ab = spd_space.metric.log(base_point=spd_mat_a, point=spd_mat_b)
        geod_ab = spd_space.metric.geodesic(
            initial_point=spd_mat_a, initial_tangent_vec=vec_ab)
        points_ab = geod_ab(np.arange(0, 1, 0.0002))
        assert is_spd(points_ab)

        vec_bc = spd_space.metric.log(base_point=spd_mat_b, point=spd_mat_c)
        geod_bc = spd_space.metric.geodesic(
            initial_point=spd_mat_b, initial_tangent_vec=vec_bc)
        points_bc = geod_bc(np.arange(0, 1, 0.0002))
        assert is_spd(points_bc)

        vec_ca = spd_space.metric.log(base_point=spd_mat_c, point=spd_mat_a)
        geod_ca = spd_space.metric.geodesic(
            initial_point=spd_mat_c, initial_tangent_vec=vec_ca)
        points_ca = geod_ca(np.arange(0, 1, 0.0002))
        assert is_spd(points_ca)
        os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

        dataset = np.concatenate([points_ab, points_bc, points_ca], axis=0)
        assert is_spd(dataset)
        # assert np.all(spd_space.belongs(dataset))
        np.random.shuffle(dataset)

        dataset = add_channels(dataset, img_dim=2)
        assert len(dataset.shape) == 4

        # dataset = normalization_linear(dataset)

        train_dataset, val_dataset = split_dataset(dataset)
        print('Saving %s...' % train_path)
        print('Saving %s...' % val_path)
        np.save(train_path, train_dataset)
        np.save(val_path, val_dataset)

    return train_dataset, val_dataset


def get_dataset_connectomes_schizophrenia():
    """
    Connectomes are SPD matrices of size N_NODESxN_NODES.
    """
    graphs = pd.read_csv('/neuro/connectomes/train_fnc.csv')
    map_functional = pd.read_csv(
        '/neuro/connectomes/comp_ind_fmri.csv', index_col=None)
    map_functional = map_functional['fMRI_comp_ind'].to_dict()
    map_functional_r = {v: k for k, v
                        in map_functional.items()}
    mapping = pd.read_csv(
        '/neuro/connectomes/rs_fmri_fnc_mapping.csv')
    graph_labels = pd.read_csv('/neuro/connectomes/train_labels.csv')
    all_graphs = [None] * N_GRAPHS
    all_labels = np.zeros(N_GRAPHS)

    def create_connectome(graph_id, mapping):
        u = np.zeros((N_NODES, N_NODES))
        nb_edges = mapping.shape[0]
        for edge in range(nb_edges):
            e0, e1 = (mapping.iloc[edge]['mapA'], mapping.iloc[edge]['mapB'])
            region0, region1 = map_functional_r[e0], map_functional_r[e1]
            corr = graphs.iloc[graph_id][edge+1]
            u[region0, region1] = corr
        u = np.multiply(u, (np.abs(u) > CORR_THRESH))
        return np.abs(u + u.T)

    for graph_id in range(N_GRAPHS):
        all_graphs[graph_id] = create_connectome(graph_id, mapping)
        all_labels[graph_id] = int(
            graph_labels.loc[graphs.index[graph_id], 'Class'])

    all_labels = np.array(all_labels)
    all_graphs = np.array(all_graphs)
    train_dataset, val_dataset = split_dataset(all_graphs)
    train_dataset = torch.Tensor(train_dataset)
    val_dataset = torch.Tensor(val_dataset)

    return train_dataset, val_dataset, all_labels


def get_dataset_cryo_sphere(img_shape_no_channel=None, kwargs=KWARGS):
    shape_str = get_shape_string(img_shape_no_channel)
    cryo_path = os.path.join(
        CRYO_TRAIN_VAL_DIR, 'cryo_%s.npy' % shape_str)

    if os.path.isfile(cryo_path):
        all_datasets = np.load(cryo_path)
    else:
        paths = glob.glob('/cryo/job40_vs_job034/*.pkl')
        all_datasets = []
        for path in paths:
            with open(path, 'rb') as pkl:
                logging.info('Loading file %s...' % path)
                data = pickle.load(pkl)
                dataset = data['ParticleStack']
                n_data = len(dataset)
                if img_shape_no_channel is not None:
                    img_h, img_w = img_shape_no_channel
                    dataset = transform.resize(
                        dataset, (n_data, img_h, img_w))

                dataset = normalization_linear(dataset)

                all_datasets.append(dataset)
        all_datasets = np.vstack([d for d in all_datasets])
        all_datasets = np.expand_dims(all_datasets, axis=1)
        np.save(cryo_path, all_datasets)

    logging.info('Cryo dataset: (%d, %d, %d, %d)' % all_datasets.shape)
    dataset = torch.Tensor(all_datasets)
    return dataset


def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]  # item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans


def get_dataset_cryo(
        dataset_name, img_shape_no_channel=None, kwargs=KWARGS):
    shape_str = get_shape_string(img_shape_no_channel)
    cryo_img_path = os.path.join(
        CRYO_TRAIN_VAL_DIR,
        '%s_%s.npy' % (dataset_name, shape_str))
    cryo_labels_path = os.path.join(
        CRYO_TRAIN_VAL_DIR,
        '%s_labels_%s.csv' % (dataset_name, shape_str))

    if os.path.isfile(cryo_img_path) and os.path.isfile(cryo_labels_path):
        all_datasets = np.load(cryo_img_path)

    else:
        if not os.path.isdir('/cryo/%s/' % dataset_name):
            os.system("cd /cryo/")
            os.system("source osf_dl_folder %s" % dataset_name)
        paths = glob.glob('/cryo/%s/final/*.h5' % dataset_name)
        all_datasets = []
        all_focuses = []
        all_thetas = []
        for path in paths:
            logging.info('Loading file %s...' % path)
            data_dict = load_dict_from_hdf5(path)
            dataset = data_dict['data']
            n_data = len(dataset)

            focus = data_dict['optics']['defocus_nominal']
            focus = np.repeat(focus, n_data)
            theta = data_dict['coordinates'][:, 3]

            if img_shape_no_channel is not None:
                img_h, img_w = img_shape_no_channel
                dataset = transform.resize(
                    dataset, (n_data, img_h, img_w))
            dataset = normalization_linear(dataset)

            all_datasets.append(dataset)
            all_focuses.append(focus)
            all_thetas.append(theta)

        all_datasets = np.vstack([d for d in all_datasets])
        all_datasets = np.expand_dims(all_datasets, axis=1)

        all_focuses = np.concatenate(all_focuses, axis=0)
        all_focuses = np.expand_dims(all_focuses, axis=1)
        all_thetas = np.concatenate(all_thetas, axis=0)
        all_thetas = np.expand_dims(all_thetas, axis=1)

        assert len(all_datasets) == len(all_focuses)
        assert len(all_datasets) == len(all_thetas)

        np.save(cryo_img_path, all_datasets)
        with open(cryo_labels_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['focus', 'theta'])
            for focus, theta in zip(all_focuses, all_thetas):
                writer.writerow([focus[0], theta[0]])
    dataset = torch.Tensor(all_datasets)
    return dataset


def get_dataset_cryo_exp(img_shape_no_channel=None, kwargs=KWARGS):
    CRYO_TRAIN_VAL_DIR = os.path.join(CRYO_DIR, 'train_val_datasets')
    shape_str = get_shape_string(img_shape_no_channel)
    cryo_img_path = os.path.join(
        CRYO_TRAIN_VAL_DIR,
        'cryo_exp_%s.npy' % shape_str)
    cryo_labels_path = os.path.join(
        CRYO_TRAIN_VAL_DIR,
        'cryo_exp_labels_%s.csv' % shape_str)

    if os.path.isfile(cryo_img_path) and os.path.isfile(cryo_labels_path):
        dataset = np.load(cryo_img_path)

    else:
        h5_path = os.path.join(CRYO_DIR, 'rib80s_sideview.h5')
        if not os.path.isfile(h5_path):
            logging.info('Downloading %s from gdrive...' % h5_path)
            os.system('cd ~')
            os.system(
                './gdrive-linux-x64 download --path %s'
                ' 11vaKEt7CIp5CdiVD3G7K4_xIeQkmZMDA' % CRYO_DIR)
        logging.info('Loading dict from %s...' % h5_path)
        data_dict = load_dict_from_hdf5(h5_path)
        dataset = data_dict['particles']
        n_data = len(dataset)

        focus = data_dict['_rlndefocusu']
        theta = data_dict['_rlnanglepsi']

        if img_shape_no_channel is not None:
            img_h, img_w = img_shape_no_channel
            dataset = transform.resize(
                dataset, (n_data, img_h, img_w))
        dataset = normalization_linear(dataset)
        dataset = np.expand_dims(dataset, axis=1)

        assert focus.shape == (n_data,), focus.shape
        assert theta.shape == (n_data,), theta.shape
        assert len(dataset) == len(focus)
        assert len(dataset) == len(theta)

        np.save(cryo_img_path, dataset)

        with open(cryo_labels_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['focus', 'theta'])
            for one_focus, one_theta in zip(focus, theta):
                writer.writerow([one_focus, one_theta])

    dataset = torch.Tensor(dataset)
    return dataset


def get_dataset_cryo_exp_class_2d(img_shape_no_channel=None,
                                  class_2d=None, kwargs=KWARGS):
    CRYO_TRAIN_VAL_DIR = os.path.join(CRYO_DIR, 'train_val_datasets')
    shape_str = get_shape_string(img_shape_no_channel)
    cryo_img_path = os.path.join(
        CRYO_TRAIN_VAL_DIR,
        'cryo_exp_class_2d_%d_%s.npy' % (class_2d, shape_str))
    cryo_labels_path = os.path.join(
        CRYO_TRAIN_VAL_DIR,
        'cryo_exp_class_2d_%d_%s_labels.csv' % (class_2d, shape_str))

    if os.path.isfile(cryo_img_path) and os.path.isfile(cryo_labels_path):
        dataset = np.load(cryo_img_path)

    else:
        h5_path = os.path.join(
            CRYO_H5,
            'class2D_%d_sort.h5' % class_2d)

        logging.info('Loading dict from %s...' % h5_path)
        data_dict = load_dict_from_hdf5(h5_path)
        dataset = data_dict['particles']
        n_data = len(dataset)

        focus = data_dict['_rlndefocusu']
        theta = data_dict['_rlnanglepsi']
        z_score = data_dict['_rlnparticleselectzscore']
        logl_contribution = data_dict['_rlnloglikelicontribution']

        if img_shape_no_channel is not None:
            img_h, img_w = img_shape_no_channel
            dataset = transform.resize(
                dataset, (n_data, img_h, img_w))
        dataset = normalization_linear(dataset)
        dataset = np.expand_dims(dataset, axis=1)

        assert focus.shape == (n_data,), focus.shape
        assert theta.shape == (n_data,), theta.shape
        assert z_score.shape == (n_data,), z_score.shape
        assert logl_contribution.shape == (n_data,), logl_contribution.shape

        np.save(cryo_img_path, dataset)

        with open(cryo_labels_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['focus', 'theta', 'z_score', 'logl_contribution'])
            for one_focus, one_theta, one_z_score, one_logl_contrib in zip(
                    focus, theta, z_score, logl_contribution):
                writer.writerow([
                    one_focus, one_theta, one_z_score, one_logl_contrib])

    dataset = torch.Tensor(dataset)
    return dataset


def get_dataset_cryo_exp_3d(img_shape_no_channel=None, kwargs=KWARGS):
    CRYO_TRAIN_VAL_DIR = os.path.join(CRYO_DIR, 'train_val_datasets')
    shape_str = get_shape_string(img_shape_no_channel)
    cryo_img_path = os.path.join(
        CRYO_TRAIN_VAL_DIR, 'cryo_exp_3d_%s.npy' % shape_str)
    if os.path.isfile(cryo_img_path):
        dataset = np.load(cryo_img_path)

    else:
        path = os.path.join(CRYO_DIR, 'data.hdf5')

        logging.info('Loading file %s...' % path)
        data_dict = load_dict_from_hdf5(path)
        dataset = data_dict['particles']

        assert img_shape_no_channel == (90, 90)

        dataset = normalization_linear(dataset)
        dataset = np.expand_dims(dataset, axis=1)

        np.save(cryo_img_path, dataset)

    dataset = torch.Tensor(dataset)
    return dataset


def get_loaders_brain(dataset_name, frac_val, batch_size,
                      img_shape, kwargs=KWARGS):

    shape_str = get_shape_string(img_shape)
    NEURO_TRAIN_VAL_DIR = os.path.join(NEURO_DIR, 'train_val_datasets')
    train_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'train_%s_%s.npy' % (dataset_name, shape_str))
    val_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'val_%s_%s.npy' % (dataset_name, shape_str))

    train = torch.Tensor(np.load(train_path))
    val = torch.Tensor(np.load(val_path))

    logging.info('-- Train tensor: (%d, %d, %d, %d)' % train.shape)
    train_dataset = torch.utils.data.TensorDataset(train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    logging.info('-- Val tensor: (%d, %d, %d, %d)' % val.shape)
    val_dataset = torch.utils.data.TensorDataset(val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, val_loader


def make_synthetic_dataset_and_decoder(synthetic_params,
                                       nn_architecture,
                                       train_params):
    """
    Generate synthetic dataset from a "true" decoder.

    The dataset is different depending on the
    vae/other submanifold learning method
    that will be used.

    - vae_type == 'gvae' or == 'gvae_tgt'
        - data generated on tangent space at base_point
        - data shot on the manifold
        - riemannian noise on the manifold
        - data log-projected on tangent space at base_point
        - data in 2D on the tangent space
    - vae_type == 'vae':
        - data generated on tangent space at base_point
        - data shot on the manifold
        - riemannian noise on the manifold
        - data in 3D in the ambient space
    """
    # TODO(nina): Unfair comparison: vae learning is not like gen model
    synthetic_dir = synthetic_params['dir']
    synthetic_data_path = os.path.join(synthetic_dir, 'dataset.npy')
    decoder_true_path = os.path.join(synthetic_dir, 'decoder_true.pth')

    synthetic_data_exists = os.path.isfile(synthetic_data_path)
    decoder_exists = os.path.isfile(decoder_true_path)

    if synthetic_data_exists and decoder_exists:
        dataset = np.load(synthetic_data_path)
    else:
        n_samples = synthetic_params['n']

        manifold_name = synthetic_params['manifold_name']
        vae_type = train_params['vae_type']
        logvarx_true = synthetic_params['logvarx_true']

        os.environ['GEOMSTATS_BACKEND'] = 'numpy'
        importlib.reload(geomstats.backend)

        logging.info('NN architecture:')
        logging.info(nn_architecture)

        logging.info('Synthetic params:')
        logging.info(synthetic_params)

        decoder_true = toynn.make_decoder_true(
            synthetic_params, nn_architecture)

        if manifold_name == 'r2':
            dataset = toynn.generate_from_decoder_fixed_var(
                decoder=decoder_true,
                logvarx=logvarx_true,
                n_samples=n_samples)

        elif manifold_name == 's2' or manifold_name == 'h2':
            if vae_type == 'gvae_tgt':
                dataset = toynn.generate_from_decoder_fixed_var_tgt(
                    decoder_true,
                    logvarx=logvarx_true,
                    n_samples=n_samples,
                    manifold_name=manifold_name)
            elif vae_type == 'vae':
                dataset = toynn.generate_from_decoder_fixed_var_riem(
                    decoder_true,
                    logvarx=logvarx_true,
                    n_samples=n_samples,
                    manifold_name=manifold_name)
            else:
                raise ValueError(vae_type)

        else:
            raise ValueError(manifold_name)

        np.save(synthetic_data_path, dataset)
        torch.save(decoder_true, decoder_true_path)
    return dataset
