from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import struct

import scipy.ndimage as nd
import scipy.io as io
from mpl_toolkits.mplot3d import Axes3D

def read_mnist_2d(filename):
    with open(filename, 'rb') as f:
        _, _, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for i in range(dims))
        result = np.fromstring(f.read(), dtype=np.uint8)
        result = result.reshape(shape)
        return result

def create_voxels(image):
    depth = 28
    voxels = np.array([image for _ in range(depth)])
    return voxels

def plot_voxels(voxels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(50)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_zlim(0, 30)
    ax.voxels(voxels,edgecolor='k')
    plt.show()


class MNIST3DDataset(Dataset):
    def __init__(self, imgs_path, labels_path):
        self.images = read_mnist_2d(os.path.join(imgs_path))
        self.labels = read_mnist_2d(os.path.join(labels_path))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.float32(self.images[idx]) / 255.
        label = np.asarray(self.labels[idx])
        voxels = create_voxels(image)
        img_tensor = torch.from_numpy(voxels)
        shape = img_tensor.shape
        img_tensor = img_tensor.view((1, shape[0], shape[1], shape[2]))
        # plot_voxels(img_tensor.squeeze().permute(1,2,0).numpy())
        label_tensor = torch.from_numpy(label).long()
        return img_tensor, label_tensor


# unit test
if __name__ == '__main__':
    imgs_path = 'data/train-images.idx3-ubyte'
    labels_path = 'data/train-labels.idx1-ubyte'
    dataset = MNIST3DDataset(imgs_path, labels_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    i, (image, label) = next(enumerate(dataloader))
#     plt.imshow(image.reshape(28, 28), cmap='gray')
#     plt.title('Label:' + str(label))
#     plt.show()
