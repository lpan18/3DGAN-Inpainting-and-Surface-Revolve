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


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class FaceDataset( Dataset ):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__( self, csv_file, root_dir, transform=None ):
        self.frame = pd.read_csv( csv_file )
        self.root_dir = root_dir
        self.transform = transform
    def __len__( self ):
        return len( self.frame )
    def __getitem__( self, idx ):
        img_name = os.path.join( self.root_dir, self.frame.iloc[ idx, 0 ] )
        image = Image.open( img_name )
        if self.transform:
            image = self.transform( image )
        note = self.frame.iloc[ idx, 1 ]
        return ( image, note )

class MaskFaceDataset( Dataset ):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__( self, csv_file, root_dir, transform=None ):
        self.frame = pd.read_csv( csv_file )
        self.root_dir = root_dir
        self.transform = transform
    def __len__( self ):
        return len( self.frame )
    def __getitem__( self, idx ):
        img_name = os.path.join( self.root_dir, self.frame.iloc[ idx, 0 ] )
        image = Image.open( img_name )
        if self.transform:
            image = self.transform( image )
        mask_name = os.path.join( self.root_dir, self.frame.iloc[ idx, 1 ] )
        mask = np.load( mask_name )
        mask = mask[ 0 :: 2, 0 :: 2 ]
        return ( image, mask )

if __name__ == '__main__':
    celebA_dataset = MaskFaceDataset('/home/csa102/gruvi/celebA/mask.csv',
                                    '/home/csa102/gruvi/celebA',
                                    transform=transforms.Compose( [
                                    transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize( ( 0.5, 0.5, 0.5 ), ( 0.5, 0.5, 0.5 ) )
                                    ] ) )
    dataloader = torch.utils.data.DataLoader( celebA_dataset,
                                              batch_size=1,
                                              shuffle=False )
    i, ( imgs, masks ) = next(enumerate( dataloader ))
    masks = np.stack( ( masks, ) * 3, axis=1 )
    corrupted = imgs * torch.tensor( masks )
    figs, axes = plt.subplots(1, 3)
    axes[0].imshow(imgs.squeeze().permute(1,2,0).numpy()) 
    axes[1].imshow(Tensor(masks.squeeze()).permute(1,2,0).cpu().numpy()) 
    axes[2].imshow(corrupted.squeeze().permute(1,2,0).numpy())
    plt.savefig('testdataloader.png')
