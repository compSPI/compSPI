import numpy as np
#from scipy.ndimage import correlate
#
def get_info(nmax):
    DATASET_DIR            = '../../../datasets/sim/randomrot1D_nodisorder_nonoise_128x128/'
    TRAIN_DATASET_PATH     = DATASET_DIR+'cryo_sim_128x128.npy'
    TRAIN_METADATASET_PATH = DATASET_DIR+'cryo_sim_labels_128x128.csv'
    dataset          = np.load(TRAIN_DATASET_PATH)
    metadata         = np.genfromtxt(TRAIN_METADATASET_PATH, delimiter=",", skip_header=1)
    return dataset[0:nmax,...], metadata[0:nmax,:]
#
def compute_neighbor_cc(defocus=2.5,nmax=10000, nsize=100):
    dataset, metadata = get_info(nmax)
    index = np.where(metadata[:,0]==defocus)[0]
    data_subset = dataset[index,0,:,:]
    meta_subset = metadata[index,1]
    index_ordered = np.argsort(meta_subset)
    #index.shape[0]
    print('nsize = {}'.format(nsize))
    cc = np.zeros(nsize)
    for i in np.arange(nsize-1):
        cc[i] = np.corrcoef(data_subset[i,0,...].flat, data_subset[i+1,0,...].flat)[0,1]
    print('<CC> = {} \ std = {}'.format(np.mean(cc),np.std(cc)))
    return cc
#
def compute_mean_cc(defocus=2.5):
    dataset, metadata = get_info()
    index = np.where(metadata[:,0]==defocus)[0]
    nsize = index.shape[0]
    print('nsize = {}'.format(nsize))
    cc = np.zeros((nsize,nsize))
    for i in np.arange(nsize):
        for j in np.arange(i+1,nsize):
            cc[i,j] = np.corrcoef(dataset[i,0,...].flat, dataset[j,0,...].flat)[0,1]
    print('<CC> = {}'.format(np.sum(cc)/(nsize*(nsize-1)/2)))
    return cc
#np.save('dataset_extended.npy',dataset_extended)
