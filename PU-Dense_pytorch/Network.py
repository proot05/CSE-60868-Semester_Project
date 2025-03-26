import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.BasicBlock import MyInception_1, Pyramid_1  # Make sure these use dense operations!

class MyNet(nn.Module):
    CHANNELS = [None, 32, 32, 64, 128, 256]
    TR_CHANNELS = [None, 32, 32, 64, 128, 256]
    BLOCK_1 = MyInception_1
    BLOCK_2 = Pyramid_1

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 bn_momentum=0.1,
                 last_kernel_size=5,
                 D=3):
        """
        This dense version of the network uses nn.Conv3d and nn.ConvTranspose3d.
        Note:
          - MinkowskiEngine’s coordinate management and sparse operations have no direct dense equivalent.
          - The generate_new_coords parameter is not available.
          - get_target_by_sp_tensor and choose_keep (which use coordinate information) are reimplemented as placeholders.
          - The pruning operation is set to a no-op.
        """
        super(MyNet, self).__init__()
        # D is assumed to be 3 (for 3D convolution)
        CHANNELS = self.CHANNELS
        TR_CHANNELS = self.TR_CHANNELS
        BLOCK_1 = self.BLOCK_1
        BLOCK_2 = self.BLOCK_2
        
        # Encoder
        self.conv1 = nn.Conv3d(in_channels, CHANNELS[1], kernel_size=5, stride=1, dilation=1,
                               bias=False, padding=5//2)
        self.norm1 = nn.BatchNorm3d(CHANNELS[1], momentum=bn_momentum)
        self.block1 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[1], bn_momentum=bn_momentum)
    
        self.conv2 = nn.Conv3d(CHANNELS[1], CHANNELS[2], kernel_size=3, stride=2, dilation=1,
                               bias=False, padding=3//2)
        self.norm2 = nn.BatchNorm3d(CHANNELS[2], momentum=bn_momentum)
        self.block2 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[2], bn_momentum=bn_momentum)
    
        self.conv3 = nn.Conv3d(CHANNELS[2], CHANNELS[3], kernel_size=3, stride=2, dilation=1,
                               bias=False, padding=3//2)
        self.norm3 = nn.BatchNorm3d(CHANNELS[3], momentum=bn_momentum)
        self.block3 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[3], bn_momentum=bn_momentum)
    
        self.conv4 = nn.Conv3d(CHANNELS[3], CHANNELS[4], kernel_size=3, stride=2, dilation=1,
                               bias=False, padding=3//2)
        self.norm4 = nn.BatchNorm3d(CHANNELS[4], momentum=bn_momentum)
        self.block4 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[4], bn_momentum=bn_momentum)
        
        self.conv5 = nn.Conv3d(CHANNELS[4], CHANNELS[5], kernel_size=3, stride=2, dilation=1,
                               bias=False, padding=3//2)
        self.norm5 = nn.BatchNorm3d(CHANNELS[5], momentum=bn_momentum)
        self.block5 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[5], bn_momentum=bn_momentum)
    
        # Decoder (Transposed Convolutions)
        self.conv5_tr = nn.ConvTranspose3d(CHANNELS[5], TR_CHANNELS[5], kernel_size=3, stride=2,
                                           dilation=1, bias=False, padding=1, output_padding=1)
        self.norm5_tr = nn.BatchNorm3d(TR_CHANNELS[5], momentum=bn_momentum)
        self.block5_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[5], bn_momentum=bn_momentum)
        
        self.conv4_tr = nn.ConvTranspose3d(CHANNELS[4] + TR_CHANNELS[5], TR_CHANNELS[4],
                                           kernel_size=3, stride=2, dilation=1, bias=False,
                                           padding=1, output_padding=1)
        self.norm4_tr = nn.BatchNorm3d(TR_CHANNELS[4], momentum=bn_momentum)
        self.block4_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[4], bn_momentum=bn_momentum)
        
        self.conv3_tr = nn.ConvTranspose3d(CHANNELS[3] + TR_CHANNELS[4], TR_CHANNELS[3],
                                           kernel_size=3, stride=2, dilation=1, bias=False,
                                           padding=1, output_padding=1)
        self.norm3_tr = nn.BatchNorm3d(TR_CHANNELS[3], momentum=bn_momentum)
        self.block3_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[3], bn_momentum=bn_momentum)
        
        # Note: generate_new_coords parameter is not applicable in dense convolution.
        self.conv2_tr = nn.ConvTranspose3d(CHANNELS[2] + TR_CHANNELS[3], TR_CHANNELS[2],
                                           kernel_size=last_kernel_size, stride=2, dilation=1,
                                           bias=False, padding=last_kernel_size//2, output_padding=1)
        self.norm2_tr = nn.BatchNorm3d(TR_CHANNELS[2], momentum=bn_momentum)
        self.block2_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[2], bn_momentum=bn_momentum)
        
        self.conv1_tr = nn.Conv3d(TR_CHANNELS[2], TR_CHANNELS[1], kernel_size=3, stride=1,
                                  dilation=1, bias=False, padding=3//2)
    
        self.final = nn.Conv3d(TR_CHANNELS[1], out_channels, kernel_size=1, stride=1,
                               dilation=1, bias=True)
        
        # There is no direct equivalent of MinkowskiPruning for dense tensors.
        self.pruning = lambda x, keep: x  # No pruning operation is applied.

    def make_layer(self, block_1, block_2, channels, bn_momentum):
        layers = []
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))
        layers.append(block_2(channels=channels, bn_momentum=bn_momentum))
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))
        return nn.Sequential(*layers)

    def get_target_by_dense_tensor(self, out, coords_T):
        """
        In the original sparse version, this function used the sparse tensor's coordinates.
        In a dense tensor formulation, such coordinate information is lost.
        This placeholder simply returns a dummy boolean tensor.
        You will need to implement an appropriate target computation.
        """
        return torch.zeros_like(out, dtype=torch.bool)

    def choose_keep(self, out, coords_T, device):
        """
        In the dense version, there is no equivalent to coordinate-based pruning.
        This placeholder returns a tensor of True values.
        """
        return torch.ones_like(out, dtype=torch.bool, device=device)

    def forward(self, x, coords_T, device, prune=True):
        """
        x is assumed to be a dense tensor of shape [B, C, D, H, W].
        The parameter coords_T (and related target and pruning operations) no longer have a direct
        dense equivalent and are handled via placeholder functions.
        """
        # Encoder
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out_s1 = self.block1(out_s1)
        out = F.relu(out_s1)
    
        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out_s2 = self.block2(out_s2)
        out = F.relu(out_s2)
    
        out_s4 = self.conv3(out)
        out_s4 = self.norm3(out_s4)
        out_s4 = self.block3(out_s4)
        out = F.relu(out_s4)
    
        out_s8 = self.conv4(out)
        out_s8 = self.norm4(out_s8)
        out_s8 = self.block4(out_s8)
        out = F.relu(out_s8)
        
        out_s16 = self.conv5(out)
        out_s16 = self.norm5(out_s16)
        out_s16 = self.block5(out_s16)
        out = F.relu(out_s16)
    
        # Decoder
        out = self.conv5_tr(out)
        out = self.norm5_tr(out)
        out = self.block5_tr(out)
        out_s8_tr = F.relu(out)
    
        # Concatenate along channel dimension
        out = torch.cat([out_s8_tr, out_s8], dim=1)
    
        out = self.conv4_tr(out)
        out = self.norm4_tr(out)
        out = self.block4_tr(out)
        out_s4_tr = F.relu(out)
    
        out = torch.cat([out_s4_tr, out_s4], dim=1)
    
        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out = self.block3_tr(out)
        out_s2_tr = F.relu(out)
    
        out = torch.cat([out_s2_tr, out_s2], dim=1)
    
        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out = self.block2_tr(out)
        out_s1_tr = F.relu(out)
        
        out = out_s1_tr + out_s1
        out = self.conv1_tr(out)
        out = F.relu(out)
        
        out_cls = self.final(out)
        target = self.get_target_by_dense_tensor(out, coords_T)
        keep = self.choose_keep(out_cls, coords_T, device)
        if prune:
            out = self.pruning(out_cls, keep)  # No pruning is applied.
        return out, out_cls, target, keep
