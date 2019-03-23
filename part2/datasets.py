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


def read_mnist_2D(filename):
    with open(filename, 'rb') as f:
        _, _, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for i in range(dims))
        result = np.fromstring(f.read(), dtype=np.uint8)
        result = result.reshape(shape)
        return result

class MNIST3DDataset(Dataset):
    def __init__(self, imgs_path, labels_path):
        self.images = read_mnist_2D(os.path.join(imgs_path))
        self.labels = read_mnist_2D(os.path.join(labels_path))
        self.shape = self.images.shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.float32(self.images[idx]) / 255.0
        label = np.asarray(self.labels[idx])
        img_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(label).long()
        return img_tensor, label_tensor


# unit test
if __name__ == '__main__':
    imgs_path = 'data/train-images.idx3-ubyte'
    labels_path = 'data/train-labels.idx1-ubyte'
    dataset = MNIST3DDataset(imgs_path, labels_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)
    i, (image, label) = next(enumerate(dataloader))
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title('Label:' + str(label))
    plt.show()

    # figs, axes = plt.subplots(1, 4)
    # for i in range(0, 4):
    #         axes[i].imshow(images[i].reshape(h, w), cmap='gray')
    #         axes[i].set_title('Label:' + str(labels[i]))
    # plt.show()
