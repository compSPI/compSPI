import numpy as np
from scipy.ndimage import rotate
#
DATASET_DIR            = '../../../datasets/sim/randomrot1D_nodisorder_nonoise_128x128/'
TRAIN_DATASET_PATH     = DATASET_DIR+'cryo_sim_128x128.npy'
TRAIN_METADATASET_PATH = DATASET_DIR+'cryo_sim_labels_128x128.csv'
dataset          = np.load(TRAIN_DATASET_PATH)
metadata         = np.genfromtxt(TRAIN_METADATASET_PATH, delimiter=",", skip_header=1)
dataset_extended = np.zeros((dataset.shape[0],4,dataset.shape[2], dataset.shape[3]))
dataset_extended[:,0:1,:,:] = dataset
#
i=0
#for angle in np.arange(15,360,15):
for angle in np.arange(90,360,90):
    print('angle = {} / dataset_extended.shape={}'.format(angle, dataset_extended.shape))
    i+=1
    #dataset_tmp = np.zeros(dataset.shape)
    #print('angle = {} / dataset_tmp.shape={}'.format(angle, dataset_tmp.shape))
    dataset_tmp = rotate(dataset,angle, axes=((2,3)), mode='wrap')
    print(dataset.shape, dataset_tmp.shape)
    dataset_extended[:,i:i+1,:,:] = dataset_tmp[:,:,0:dataset.shape[2],0:dataset.shape[3]] #rotate(dataset,angle, axes=((2,3)), mode='wrap')
#
np.save('dataset_extended.npy',dataset_extended)
