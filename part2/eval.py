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
    #for i in range(20):
    fig = plt.figure()
    ax = fig.gca( projection='3d' )
    ax.view_init(60, 300)
    ax.set_aspect(0.7)
    obj = (voxels[0] > 0.3)
    ax.voxels(obj, edgecolor='k')
    #fig.savefig('completion/' + text + '_' + str(i) + '.png')
    fig.savefig('completion/' + text + '.png')
    plt.close()

def main():
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    load_models = [9600,12400,12800,16000,18000,18800,21600,24000,28000,28800,30400,30800,32000,33200,34000,35600,36000]
    for load_model in load_models:
      generator = torch.load( 'models/gen_' + str(load_model) + '.pt' )
      if cuda:
          generator.cuda()
      generator.eval()
      os.makedirs( 'completion', exist_ok=True )
      latent_dim = 100
      nums = 7
      
      interpolate = True
      test = False
      
      if test:
        z = Variable( Tensor( np.random.normal( 0, 1, (100, latent_dim) ) ) )
        gen_imgs = generator(z)
        save_image( gen_imgs.detach().cpu().numpy(), str(load_model) )
        
      if interpolate:
        for i in range(5):
            z1 = np.random.normal(0, 1, (latent_dim, ))
            z2 = np.random.normal(0, 1, (latent_dim, ))
            z = np.zeros((latent_dim,nums))
            for idx in range(latent_dim):
                z[idx] = np.linspace(z1[idx], z2[idx], nums)
            z = np.transpose(z)
            z = Variable(Tensor(z)) # 10*100
            gen_imgs = generator(z)
            for j in range(nums):
                save_image(gen_imgs[j].detach().cpu().numpy(),  str(i) + '_' + str(j))


if __name__ == '__main__':
    main()
