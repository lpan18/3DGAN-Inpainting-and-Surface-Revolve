import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from scipy.signal import convolve2d
import external.poissonblending as blending
import numpy as np
import pdb

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class ContextLoss( nn.Module ):
    def __init__( self, generated ):
        super( ContextLoss, self ).__init__()
        self.generated = generated.detach()

    def forward( self, corrupted, weight_mask ):
        loss = F.l1_loss( corrupted, self.generated, reduction='none' ) * weight_mask
        loss = torch.mean( loss )
        return loss

class PriorLoss( nn.Module ):
    def __init__( self ):
        super( PriorLoss, self ).__init__()
        self.normal = torch.distributions.normal.Normal( Tensor( ( 0., ) ),
                                                         Tensor( ( 1., ) ) )

    def forward( self, z ):
        loss = self.normal.log_prob( z )
        loss = -torch.sum( loss )
        return loss


class ModelInpaint():
    def __init__( self, args ):
        self.batch_size = args.batch_size
        self.z_dim = 100
        self.n_size = args.n_size
        self.per_iter_step = args.per_iter_step
        self.prior_likelihood = args.prior_likelihood

        self.generator = torch.load( args.generator )
        self.generator.eval()
        self.discriminator = torch.load( args.discriminator )
        self.discriminator.eval()

    def create_weight_mask_hole( self, unweighted_masks, decay_index=0 ):
        kernel = np.ones( ( self.n_size, self.n_size ),
                          dtype=np.float32 )
        kernel = kernel / np.sum( kernel )
        weight_masks = np.zeros( unweighted_masks.shape, dtype=np.float32 )
        for i in range( weight_masks.shape[ 0 ] ):
            for j in range( weight_masks.shape[ 1 ] ):
                weight_masks[ i, j ] = convolve2d( unweighted_masks[ i, j ],
                                                   kernel,
                                                   mode='same',
                                                   boundary='symm' )
        weight_masks = unweighted_masks * ( 1.0 - weight_masks )
        return Tensor( weight_masks )

    def postprocess( self, corrupted, masks, generated ):
        corrupted = corrupted * 0.5 + 0.5
        generated = generated * 0.5 + 0.5
        corrupted = corrupted.permute( 0, 3, 2, 1 ).cpu().numpy()
        processed = generated.permute( 0, 3, 2, 1 ).cpu().detach().numpy()
        masks = np.transpose( masks, axes=( 0, 3, 2, 1 ) )
        for i in range( len( processed ) ):
            processed[ i ] = blending.blend( corrupted[ i ],
                                             processed[ i ],
                                             1 - masks[ i ] )
        processed = torch.tensor( processed ).permute( 0, 3, 2, 1 )
        return ( processed * 2.0 - 1.0 ).cuda()

    def inpaint( self, corrupted, masks, decay_index=0 ):
        z = torch.tensor( np.float32( np.random.randn( self.batch_size,
                                                       self.z_dim ) ) )
        weight_mask = self.create_weight_mask_hole( masks,
                                                         decay_index=decay_index )
        if cuda:
            z = z.cuda()
            corrupted = corrupted.cuda()
            weight_mask = weight_mask.cuda()
        optimizer = optim.Adam( [ z.requires_grad_() ] )
        #optimizer = optim.SGD( [ z.requires_grad_() ], lr=5.0 )
        print( 'Before optimizing: ' )
        print( z )
        for i in range( self.per_iter_step ):
            l1_loss = nn.MSELoss()#nn.L1Loss()
            def closure():
                optimizer.zero_grad()
                generated = self.generator( z )
                context_loss = ContextLoss( generated )( corrupted, weight_mask )
                if self.prior_likelihood:
                    prior_loss = PriorLoss()( z )
                    loss = context_loss + 0.003 * prior_loss
                else:
                    valid = Variable( Tensor( self.batch_size, 1 ).fill_( 1.0 ),
                                      requires_grad=False )
                    prior_loss = nn.BCELoss()( self.discriminator( generated ),
                                               valid )
                    loss = 10.0 * context_loss + 1.0 * prior_loss
                print( 'Iteration %d:' % i )
                print( 'Context loss: %.2f' % torch.mean( context_loss ) )
                print( 'Prior loss: %.2f' % torch.mean( prior_loss ) )
                loss.backward()# retain_graph=True )
                return loss
            optimizer.step( closure )
            # z = torch.clamp( z, -1.0, 1.0 )
        print( 'After optimizing: ' )
        print( z )
        generated = self.generator( z )
        return generated, self.postprocess( corrupted, masks, generated )
 