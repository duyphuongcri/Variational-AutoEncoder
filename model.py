"""
Author: Duy-Phuong Dao
Email : phuongdd.1997@gmail.com or duyphuongcri@gmail.com
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math 
import group_norm
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride=1, p=1, num_groups=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            #nn.GroupNorm(num_groups=num_groups, num_channels=ch_in),
            # group_norm.GroupNorm3d(num_features=ch_in, num_groups=num_groups),
            # nn.ReLU(inplace=True), 
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=p),  
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.conv(x)
        return out


class ResNet_block(nn.Module):
    "A ResNet-like block with the GroupNorm normalization providing optional bottle-neck functionality"
    def __init__(self, ch, k_size, stride=1, p=1, num_groups=1):
        super(ResNet_block, self).__init__()
        self.conv = nn.Sequential(
            #nn.GroupNorm(num_groups=num_groups, num_channels=ch),
            # group_norm.GroupNorm3d(num_features=ch, num_groups=num_groups),
            # nn.ReLU(inplace=True), 
            nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p), 
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),

            #nn.GroupNorm(num_groups=num_groups, num_channels=ch),
            # group_norm.GroupNorm3d(num_features=ch, num_groups=num_groups),
            # nn.ReLU(inplace=True), 
            nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p),  
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.conv(x) + x
        return out


class up_conv(nn.Module):
    "Reduce the number of features by 2 using Conv with kernel size 1x1x1 and double the spatial dimension using 3D trilinear upsampling"
    def __init__(self, ch_in, ch_out, k_size=1, scale=2, align_corners=False):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size),
            nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=align_corners),
        )
    def forward(self, x):
        return self.up(x)

class Encoder(nn.Module):
    """ Encoder module """
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = conv_block(ch_in=1, ch_out=32, k_size=3, num_groups=1)
        self.res_block1 = ResNet_block(ch=32, k_size=3, num_groups=8)
        self.MaxPool1 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv2 = conv_block(ch_in=32, ch_out=64, k_size=3, num_groups=8)
        self.res_block2 = ResNet_block(ch=64, k_size=3, num_groups=16)
        self.MaxPool2 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv3 = conv_block(ch_in=64, ch_out=128, k_size=3, num_groups=16)
        self.res_block3 = ResNet_block(ch=128, k_size=3, num_groups=16)
        self.MaxPool3 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv4 = conv_block(ch_in=128, ch_out=256, k_size=3, num_groups=16)
        self.res_block4 = ResNet_block(ch=256, k_size=3, num_groups=16)
        self.MaxPool4 = nn.MaxPool3d(3, stride=2, padding=1)

        self.reset_parameters()
      
    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.res_block1(x1)
        x1 = self.MaxPool1(x1) # torch.Size([1, 32, 26, 31, 26])
        
        x2 = self.conv2(x1)
        x2 = self.res_block2(x2)
        x2 = self.MaxPool2(x2) # torch.Size([1, 64, 8, 10, 8])

        x3 = self.conv3(x2)
        x3 = self.res_block3(x3)
        x3 = self.MaxPool3(x3) # torch.Size([1, 128, 2, 3, 2])
        
        x4 = self.conv4(x3)
        x4 = self.res_block4(x4) # torch.Size([1, 256, 2, 3, 2])
        x4 = self.MaxPool4(x4) # torch.Size([1, 256, 1, 1, 1])
        # print("x1 shape: ", x1.shape)
        # print("x2 shape: ", x2.shape)
        # print("x3 shape: ", x3.shape)
        # print("x4 shape: ", x4.shape) 
        return x4

class Decoder(nn.Module):
    """ Decoder Module """
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.linear_up = nn.Linear(latent_dim, 256*150)
        self.relu = nn.ReLU()
        self.upsize4 = up_conv(ch_in=256, ch_out=128, k_size=1, scale=2)
        self.res_block4 = ResNet_block(ch=128, k_size=3, num_groups=16)
        self.upsize3 = up_conv(ch_in=128, ch_out=64, k_size=1, scale=2)
        self.res_block3 = ResNet_block(ch=64, k_size=3, num_groups=16)        
        self.upsize2 = up_conv(ch_in=64, ch_out=32, k_size=1, scale=2)
        self.res_block2 = ResNet_block(ch=32, k_size=3, num_groups=16)   
        self.upsize1 = up_conv(ch_in=32, ch_out=1, k_size=1, scale=2)
        self.res_block1 = ResNet_block(ch=1, k_size=3, num_groups=1)   

        self.reset_parameters()
      
    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        x4_ = self.linear_up(x)
        x4_ = self.relu(x4_)

        x4_ = x4_.view(-1, 256, 5, 6, 5)
        x4_ = self.upsize4(x4_) 
        x4_ = self.res_block4(x4_)

        x3_ = self.upsize3(x4_) 
        x3_ = self.res_block3(x3_)

        x2_ = self.upsize2(x3_) 
        x2_ = self.res_block2(x2_)

        x1_ = self.upsize1(x2_) 
        x1_ = self.res_block1(x1_)

        return x1_


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.latent_dim = latent_dim
        self.z_mean = nn.Linear(256*150, latent_dim)
        self.z_log_sigma = nn.Linear(256*150, latent_dim)
        self.epsilon = torch.normal(size=(1, latent_dim), mean=0, std=1.0, device=self.device)
        self.encoder = Encoder()
        self.decoder = Decoder(latent_dim)

        self.reset_parameters()
      
    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        z_mean = self.z_mean(x)
        z_log_sigma = self.z_log_sigma(x)
        z = z_mean + z_log_sigma.exp()*self.epsilon
        y = self.decoder(z)
        return y, z_mean, z_log_sigma
