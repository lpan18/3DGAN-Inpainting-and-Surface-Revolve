from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import struct
from skimage.transform import resize
from mpl_toolkits.mplot3d import Axes3D

def read_mnist_2d(filename):
    # Reference: 
    # https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    with open(filename, 'rb') as f:
        _, _, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for i in range(dims))
        result = np.fromstring(f.read(), dtype=np.uint8)
        result = result.reshape(shape)
        return result

def create_voxels(image):
    depth = 15
    voxels = np.zeros((image.shape[0],image.shape[0],image.shape[0]))
    for idx in range(depth):
        voxels[:,:,5+idx] = image
    return voxels

def plot_voxels(voxels, label):
    fig = plt.figure()
    ax = fig.gca( projection='3d' )
    ax.view_init(60, 300)
    voxels = (voxels[0,0] > 0.5)
    ax.voxels(voxels,edgecolor='k')
    # plt.show()
    fig.savefig( 'test_dataloader_%s.png' % label[0].numpy() )
    plt.close()
        
class MNIST3DDataset(Dataset):
    def __init__(self, imgs_path, labels_path):
        self.images = read_mnist_2d(os.path.join(imgs_path))
        self.labels = read_mnist_2d(os.path.join(labels_path))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.float32(self.images[idx]) / 255.0
        image = resize(image, (32, 32))
        label = np.asarray(self.labels[idx])
        voxels = create_voxels(image)
        img_tensor = torch.from_numpy(voxels)
        shape = img_tensor.shape
        img_tensor = img_tensor.view((1, shape[0], shape[1], shape[2]))
        label_tensor = torch.from_numpy(label).long()
        return img_tensor, label_tensor


# unit test
if __name__ == '__main__':
    imgs_path = 'data/train-images.idx3-ubyte'
    labels_path = 'data/train-labels.idx1-ubyte'
    dataset = MNIST3DDataset(imgs_path, labels_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    for i, (image, label) in enumerate(dataloader):
        plot_voxels(image.detach().cpu().numpy(), label)
        if(i==5):
            break
