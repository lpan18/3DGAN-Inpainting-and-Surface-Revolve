from PIL import Image
import argparse
import os
import numpy as np
import torch
from datasets import MaskFaceDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from glob import glob
import pdb
from model import ModelInpaint
from dcgan import Generator, Discriminator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--generator',
                         type=str,
                         help='Pretrained generator',
                         default='/home/csa102/CMPT743/PyTorch-GAN/implementations/semantic_image_inpainting/models/gen_9600.pt' )
    parser.add_argument( '--discriminator',
                         type=str,
                         help='Pretrained discriminator',
                         default='/home/csa102/CMPT743/PyTorch-GAN/implementations/semantic_image_inpainting/models/dis_9600.pt' )
    parser.add_argument( '--imgSize',
                         type=int,
                         default=64 )
    parser.add_argument( '--batch_size',
                         type=int,
                         default=64 )
    parser.add_argument( '--n_size',
                         type=int,
                         default=7,
                         help='size of neighborhood' )
    parser.add_argument( '--blend',
                         action='store_true',
                         default=True,
                         help="Blend predicted image to original image" )
    # These files are already on the VC server. Not sure if students have access to them yet.
    parser.add_argument( '--mask_csv',
                         type=str,
                         default='/home/csa102/gruvi/celebA/mask.csv',
                         help='path to the masked csv file' )
    parser.add_argument( '--test_csv',
                         type=str,
                         default='/home/csa102/gruvi/celebA/test.csv',
                         help='path to the test csv file' )
    parser.add_argument( '--mask_root',
                         type=str,
                         default='/home/csa102/gruvi/celebA',
                         help='path to the masked root' )
    parser.add_argument( '--per_iter_step',
                         type=int,
                         default=1500,
                         help='number of steps per iteration' )
    args = parser.parse_args()
    return args

def saveimages( corrupted, completed, blended, index ):
    os.makedirs( 'completion', exist_ok=True )
    save_image( corrupted,
                'completion/%d_corrupted.png' % index,
                nrow=corrupted.shape[ 0 ] // 5,
                normalize=True )
    save_image( completed,
                'completion/%d_completed.png' % index,
                nrow=completed.shape[ 0 ] // 5,
                normalize=True )
    save_image( blended,
                'completion/%d_blended.png' % index,
                nrow=corrupted.shape[ 0 ] // 5,
                normalize=True )

def test():
    args = parse_args()
    m = ModelInpaint( args )

    img_name = 'selfie.jpg'
    mask_name = '/home/csa102/gruvi/celebA/mask/180000_mask.npy'
    img_path = os.path.join(img_name)
    mask_path = os.path.join(mask_name)

    transform = transforms.Compose( [
            transforms.Resize( args.imgSize ),
            transforms.ToTensor(),
            transforms.Normalize( ( 0.5, 0.5, 0.5 ), ( 0.5, 0.5, 0.5 ) )
            ] ) 

    image = Image.open(img_path)
    image = transform(image)

    mask = np.load(mask_path)
    mask = mask[ 0 :: 2, 0 :: 2 ]
    mask = np.stack((mask,) * 3, axis=1 )

    corrupted = imgs * torch.tensor(mask)
    completed, blended = m.inpaint(corrupted, mask)
    
    save_image( corrupted, 'completion/selfie_corrupted.png',)
    save_image( completed, 'completion/selfie_completed.png',)
    save_image( blended, 'completion/selfie_blended.png',)

def main():
    args = parse_args()
    # Configure data loader
    celebA_dataset = MaskFaceDataset( args.mask_csv,
                                      args.mask_root,
                                      transform=transforms.Compose( [
                           transforms.Resize( args.imgSize ),
                           transforms.ToTensor(),
                           transforms.Normalize( ( 0.5, 0.5, 0.5 ), ( 0.5, 0.5, 0.5 ) )
                       ] ) )
    dataloader = torch.utils.data.DataLoader( celebA_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False )
    m = ModelInpaint( args )
    for i, ( imgs, masks ) in enumerate( dataloader ):
        masks = np.stack( ( masks, ) * 3, axis=1 )
        corrupted = imgs * torch.tensor( masks )
        completed, blended = m.inpaint( corrupted, masks )
        saveimages( corrupted, completed, blended, i )
        corrupted = blended


if __name__ == '__main__':
    main()
