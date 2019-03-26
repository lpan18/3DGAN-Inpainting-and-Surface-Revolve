import argparse
import os
import numpy as np
from torch.autograd import Variable
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dcgan import Generator

parser = argparse.ArgumentParser()

def save_image(voxels, text):
    fig = plt.figure()
    ax = fig.gca( projection='3d' )
    ax.view_init(60, 300)
    ax.set_aspect(0.7)
    obj = (voxels[0].squeeze().permute(1,2,0).numpy() > 0.5)
    ax.voxels(obj, edgecolor='k')
    fig.savefig('completion/' + text + '.png')
    plt.close()

def main():
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    generator = torch.load( 'models/gen_36000.pt' )
    if cuda:
        generator.cuda()
    generator.eval()
    os.makedirs( 'completion', exist_ok=True )
    latent_dim = 100
    nums = 10
    for i in range(nums):
        z1 = np.random.normal(0, 1, (latent_dim, ))
        z2 = np.random.normal(0, 1, (latent_dim, ))
        z = np.zeros((latent_dim,nums))
        for idx in range(latent_dim):
            z[idx] = np.linspace(z1[idx], z2[idx], nums)
        z = np.transpose(z)
        z = Variable(Tensor(z)) # 10*100
        gen_imgs = generator(z)
        for j in range(nums):
            save_image(gen_imgs[j].cpu().detach(),  str(i) + '_' + str(j))

if __name__ == '__main__':
    main()
