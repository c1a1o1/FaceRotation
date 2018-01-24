import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class residual_block(nn.Module):
    def __init__(self, channels):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return out

#the Generator contains only one global part if TwoPathway = false, vice versa
class Generator(nn.Module):
    def __init__(self, channel_num = 3, num_classes = 1200, TwoPathway = False, nz = 100):
        super(Generator, self).__init__()
        self.TwoPathway = TwoPathway

        #Encoder
        #128 x 128 x 64
        self.conv0 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size = 7, stride = 1, padding = 3),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace = True),
                    residual_block(64)
            )

        #64 x 64 x 64
        self.conv1 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size = 5, stride = 2, padding = 2),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace = True),
                    residual_block(64)
                )

        #32 x 32 x 128
        self.conv2 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace = True),
                    residual_block(128)
                )

        #16 x 16 x 256
        self.conv3 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace = True),
                    residual_block(256)
                )

        #8 x 8 x 512
        self.conv4 = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace = True),
                    residual_block(512),
                    residual_block(512),
                    residual_block(512),
                    residual_block(512)
                )

        #fc1 512
        self.fc1 = nn.Sequential(
                    nn.Linear(32768, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2, inplace = True)
                )

        #extra softmax for classification, and it also help to extract the feature of the profile image
        self.extra = nn.Sequential(
                    nn.Linear(256, num_classes)
                )

        #Decoder
        #First Part, simply deconvolution
        #FC 8 x 8 x 64
        self.feat8_1 = nn.Linear(256 + nz, 4096)
        self.feat8_2 = nn.Sequential(
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace = True),
                    residual_block(64)
                )
        
        #32 x 32 x 32
        self.feat32 = nn.Sequential(
                    nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 4, output_padding = 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace = True),
                    residual_block(32)
                )
        
        #64 x 64 x 16
        self.feat64 = nn.Sequential(
                    nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace = True),
                    residual_block(16)
                )
        
        #128 x 128 x 8
        self.feat128 = nn.Sequential(
                    nn.ConvTranspose2d(16, 8, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(inplace = True),
                    residual_block(8),
                    residual_block(8)
                )
        
        #Second Part
        #16 x 16 x 512
        self.deconv0 = nn.Sequential(
                    nn.ConvTranspose2d(64 + 512, 512, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace = True)
                )
        
        #32 x 32 x 256
        self.deconv1 = nn.Sequential(
                    nn.ConvTranspose2d(512 + 256, 256, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace = True)
                )
        
        #64 x 64 x 128
        self.deconv2 = nn.Sequential(
                    nn.ConvTranspose2d(256 + 32 + 128 + 3, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace = True)
                )
        
        #128 x 128 x 64
        self.deconv3 = nn.Sequential(
                    nn.ConvTranspose2d(128 + 16 + 64 + 3, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace = True)
                )
        
        #128 x 128 x 64
        self.conv5 = nn.Sequential(
                    nn.Conv2d(64 + 8 + 64 + (3 if TwoPathway else 0) + 3, 64, kernel_size = 5, stride = 1, padding = 2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace = True)
                )
        
        #128 x 128 x 32
        self.conv6 = nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size = 3, stride = 1, padding = 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace = True)
                )
        
        #128 x 128 x 3
        self.conv7 = nn.Sequential(
                    nn.Conv2d(32, 3, kernel_size = 3, stride = 1, padding = 1),
                    nn.Tanh()
                )

    def forward(self, x128, x64, x32, z, local = None):
        #Encoder
        conv0 = self.conv0(x128)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        fc = self.fc1(conv4.view(-1, 32768))
        fc = torch.split(fc, 256, 1)
        fc = torch.max(fc[0], fc[1])
        extra = self.extra(fc)

        #Decoder
        #concatenate fc and noise z for the input of self.feat8
        feat8 = self.feat8_1(torch.cat((fc, z), dim = 1))
        feat8 = feat8.view(-1, 64, 8, 8)
        feat8 = self.feat8_2(feat8)
	
        feat32 = self.feat32(feat8)
        feat64 = self.feat64(feat32)
        feat128 = self.feat128(feat64)
        #concatenate feat8 and conv4
        out = self.deconv0(torch.cat((feat8, conv4), dim = 1))
        #concatenate out and conv3
        out = self.deconv1(torch.cat((out, conv3), dim = 1))
        #concatenate out, feat32, conv2 and x32
        out = self.deconv2(torch.cat((out, feat32, conv2, x32), dim = 1))
        #concatenate out, feat64, conv1 and x64
        out = self.deconv3(torch.cat((out, feat64, conv1, x64), dim = 1))

        #concatenate out, feat128, conv0, x if TwoPathway = false else plus local
        if not self.TwoPathway:
            out = self.conv5(torch.cat((out, feat128, conv0, x128), dim = 1))
        else:
            out = self.conv5(torch.cat((out, feat128, conv0, local, x128), dim = 1))

        out = self.conv6(out)
        out = self.conv7(out)
        return out, extra

#the local part of the network
#the input of the two eyes is w = 40, h = 40
#the input of the nose     is w = 40, h = 32
#the input pf the mouth    is w = 48, h = 32
#the size of input is w x h, the same as output
class Local(nn.Module):
    def __init__(self):
        super(Local, self).__init__()

        #w x h x 64
        self.conv0 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace = True),
                    residual_block(64),
                )

        #w/2 x h/2 x 128
        self.conv1 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding =  1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace = True),
                    residual_block(128),
                )

        #w/4 x h/4 x 256
        self.conv2 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace = True),
                    residual_block(256),
                )
        
        #w/8 x h/8 x 512
        self.conv3 = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace = True),
                    residual_block(512),
                    residual_block(512),
                )
        
        #w/4 x h/4 x 256
        self.deconv0 = nn.Sequential(
                    nn.ConvTranspose2d(512, 256, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace = True)
                )

        #w/2 x h/2 x 128
        self.deconv1 = nn.Sequential(
                    nn.ConvTranspose2d(512, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace = True)
                )

        #w x h x 64
        self.deconv2 = nn.Sequential(
                    nn.ConvTranspose2d(256, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace = True)
                )

        #w x h x 64
        self.conv4 = nn.Sequential(
                    nn.Conv2d(128, 64, kernel_size = 3, stride = 1, padding = 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace = True)
                )

        #w x h x 3
        self.conv5 = nn.Sequential(
                    nn.Conv2d(64, 3, kernel_size = 3, stride = 1, padding = 1),
                    nn.Tanh(),
                )

    def forward(self, x):
        #Encoder
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        out = self.conv3(conv2)
        
        #Decoder
        out = self.deconv0(out)
        #concatenate out and conv2
        out = self.deconv1(torch.cat((out, conv2), dim = 1))
        #concatenate out and conv1
        out = self.deconv2(torch.cat((out, conv1), dim = 1))
        #concatenate out and conv3
        out = self.conv4(torch.cat((out, conv0), dim = 1))
        out = self.conv5(out)
        return out

#this structure refers to DR-GAN, because the paper don't contains any detail of this network
#in paper, the ouput of this network is 2 x 2 probability map instead of one scalar value,
#originally I use average pooling to get 2 x 2 output, however, I think it's not that important,
#Each probability value conresponds to a certain semantic region.
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        layer = [
            #128 x 128 x 32
            nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            #128 x 128 x 64
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            #64 x 64 x 64
            nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            #64 x 64 x 64
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            #64 x 64 x 128
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            #32 x 32 x 128
            nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            #32 x 32 x 96
            nn.Conv2d(128, 96, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(96),
            nn.ELU(),
            #32 x 32 x 192
            nn.Conv2d(96, 192, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(192),
            nn.ELU(),
            #16 x 16 x 192
            nn.Conv2d(192, 192, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(192),
            nn.ELU(),
            #16 x 16 x 128
            nn.Conv2d(192, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            #16 x 16 x 256
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            #8 x 8 x 256
            nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            #8 x 8 x 160
            nn.Conv2d(256, 160, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(160),
            nn.ELU(),
            #8 x 8 x 320
            nn.Conv2d(160, 320, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(320),
            nn.ELU(),
            #AvgPool 2 x 2 x 320
            nn.AvgPool2d(kernel_size = 4, stride = 4),
            nn.BatchNorm2d(320),
            nn.ELU(),
            #2 x 2,
            nn.Conv2d(320, 1, kernel_size = 1, stride = 1),
            nn.Sigmoid(),
        ]
        self.layer = nn.Sequential(*layer)
        
    def forward(self, x):
        x = self.layer(x)
        x = x.view(-1, 2, 2)
        return x
