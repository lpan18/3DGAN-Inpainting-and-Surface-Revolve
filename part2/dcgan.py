# Reference: 
# https://github.com/eriklindernoren/Keras-GAN
# https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import MNIST3DDataset
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import pdb

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument( '--n_epochs',
                     type=int,
                     default=20,
                     help='number of epochs of training' )
parser.add_argument( '--batch_size',
                     type=int,
                     default=32,
                     help='size of the batches' )
parser.add_argument( '--lr',
                     type=float,
                     default=0.0002,
                     help='adam: learning rate' )
parser.add_argument( '--b1',
                     type=float,
                     default=0.5,
                     help='adam: decay of first order momentum of gradient' )
parser.add_argument( '--b2',
                     type=float,
                     default=0.999,
                     help='adam: decay of first order momentum of gradient' )
parser.add_argument( '--n_cpu',
                     type=int,
                     default=8,
                     help='number of cpu threads to use during batch generation' )
parser.add_argument( '--latent_dim',
                     type=int,
                     default=100,
                     help='dimensionality of the latent space' )
parser.add_argument( '--obj_size',
                     type=int,
                     default=32,
                     help='size of each dimension' )
parser.add_argument( '--channels',
                     type=int,
                     default=1,
                     help='number of image channels' )
parser.add_argument( '--sample_interval',
                     type=int,
                     default=400,
                     help='interval between image sampling' )
parser.add_argument( '--train_images',
                     type=str,
                     default='data/train-images.idx3-ubyte',
                     help='path to the training images' )
parser.add_argument( '--train_labels',
                     type=str,
                     default='data/train-labels.idx1-ubyte',
                     help='path to the training labels' )
opt = parser.parse_args()

class Generator( nn.Module ):
    def __init__( self, d=64 ):
        super( Generator, self ).__init__()
        self.deconv1 = nn.ConvTranspose3d( opt.latent_dim, d * 4, 4, 1, 0 )
        self.deconv1_bn = nn.BatchNorm3d( d * 4 )
        self.deconv2 = nn.ConvTranspose3d( d * 4, d * 2, 4, 2, 1 )
        self.deconv2_bn = nn.BatchNorm3d( d * 2 )
        self.deconv3 = nn.ConvTranspose3d( d * 2, d, 4, 2, 1 )
        self.deconv3_bn = nn.BatchNorm3d( d )
        self.deconv4 = nn.ConvTranspose3d( d, 1, 4, 2, 1 )

    # weight_init
    def weight_init( self, mean, std ):
        for m in self._modules:
            normal_init( self._modules[ m ], mean, std )

    # forward method
    def forward( self, input ):
        x = input.view( -1, opt.latent_dim, 1, 1, 1 )
        x = F.relu( self.deconv1_bn( self.deconv1( x ) ) )
        x = F.relu( self.deconv2_bn( self.deconv2( x ) ) )
        x = F.relu( self.deconv3_bn( self.deconv3( x ) ) )
        x = torch.tanh( self.deconv4( x ) )
        return x

class Discriminator( nn.Module ):
    # initializers
    def __init__( self, d=64 ):
        super( Discriminator, self ).__init__()
        self.conv1 = nn.Conv3d( 1, d, 4, 2, 1 )
        self.conv2 = nn.Conv3d( d, d * 2, 4, 2, 1 )
        self.conv2_bn = nn.BatchNorm3d( d * 2 )
        self.conv3 = nn.Conv3d( d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm3d( d * 4 )
        self.conv4 = nn.Conv3d( d * 4, 1, 4, 1, 0 )

    # weight_init
    def weight_init( self, mean, std ):
        for m in self._modules:
            normal_init( self._modules[ m ], mean, std )

    # forward method
    def forward( self, input ):
        x = input.view( -1, 1, opt.obj_size, opt.obj_size, opt.obj_size )
        x = F.leaky_relu( self.conv1( x ), 0.2 )
        x = F.leaky_relu( self.conv2_bn( self.conv2( x ) ), 0.2 )
        x = F.leaky_relu( self.conv3_bn( self.conv3( x ) ), 0.2 )
        x = torch.sigmoid( self.conv4( x ) )
        return x

def normal_init( m, mean, std ):
    if isinstance( m, nn.ConvTranspose3d ) or isinstance( m, nn.Conv3d ):
        m.weight.data.normal_( mean, std )
        m.bias.data.zero_()

def save_image(voxels, text):
    for i in range(5):
        fig = plt.figure()
        ax = fig.gca( projection='3d' )
        ax.view_init(60, 300)
        ax.set_aspect(0.7)
        obj = (voxels[i,0] > 0.5)
        ax.voxels(obj, edgecolor='k')
        fig.savefig('images/' + text + '_' + str(i) + '.png')
        plt.close()

def main():
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    # Initialize weights
    generator.weight_init( mean=0.0, std=0.02 )
    discriminator.weight_init( mean=0.0, std=0.02 )
    # Configure data loader
    MNIST_3D_dataset = MNIST3DDataset( opt.train_images, opt.train_labels)
    dataloader = torch.utils.data.DataLoader( MNIST_3D_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=True, 
                                              num_workers=6 )
    # Optimizers
    optimizer_G = torch.optim.Adam( generator.parameters(),
                                    lr=opt.lr,
                                    betas=( opt.b1, opt.b2 ) )
    optimizer_D = torch.optim.Adam( discriminator.parameters(),
                                    lr=opt.lr,
                                    betas=( opt.b1, opt.b2 ) )
    # ----------
    #  Training
    # ----------
    os.makedirs( 'images', exist_ok=True )
    os.makedirs( 'models', exist_ok=True )
    for epoch in range( opt.n_epochs ):
        # learning rate decay
        if ( epoch + 1 ) == 11:
            optimizer_G.param_groups[ 0 ][ 'lr' ] /= 10
            optimizer_D.param_groups[ 0 ][ 'lr' ] /= 10
            print( 'learning rate change!' )
        if ( epoch + 1 ) == 16:
            optimizer_G.param_groups[ 0 ][ 'lr' ] /= 10
            optimizer_D.param_groups[ 0 ][ 'lr' ] /= 10
            print( 'learning rate change!' )
        for i, ( imgs, _ ) in enumerate( dataloader ):
            # Adversarial ground truths
            valid = Variable( Tensor( imgs.shape[ 0 ], 1 ).fill_( 1.0 ),
                              requires_grad=False )
            fake = Variable( Tensor( imgs.shape[ 0 ], 1 ).fill_( 0.0 ),
                             requires_grad=False )
            # Configure input
            real_imgs = Variable( imgs.type( Tensor ) )
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            # Sample noise as generator input
            z = Variable( Tensor( np.random.normal( 0, 1, ( imgs.shape[ 0 ],
                                                            opt.latent_dim ) ) ) )
            # Generate a batch of images
            gen_imgs = generator( z ) 
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss( discriminator( gen_imgs ), valid )
            g_loss.backward()
            optimizer_G.step()
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            label_real = discriminator( real_imgs )
            label_gen = discriminator( gen_imgs.detach() )
            real_loss = adversarial_loss( label_real, valid )
            fake_loss = adversarial_loss( label_gen, fake )
            d_loss = ( real_loss + fake_loss ) / 2
            real_acc = ( label_real > 0.5 ).float().sum() / real_imgs.shape[ 0 ]
            gen_acc = ( label_gen < 0.5 ).float().sum() / gen_imgs.shape[ 0 ]
            d_acc = ( real_acc + gen_acc ) / 2
            d_loss.backward()
            optimizer_D.step()
            print( "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % \
                    ( epoch,
                      opt.n_epochs,
                      i,
                      len(dataloader),
                      d_loss.item(),
                      d_acc * 100,
                      g_loss.item() ) )
            batches_done = epoch * len( dataloader ) + i
            if batches_done % opt.sample_interval == 0:
                save_image( gen_imgs.detach().cpu().numpy(), str(batches_done) )
                torch.save( generator, 'models/gen_%d.pt' % batches_done )
                torch.save( discriminator, 'models/dis_%d.pt' % batches_done )
if __name__ == '__main__':
    main()
