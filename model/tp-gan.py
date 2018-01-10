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

class reshape(nn.Module):
    def __init__(self, *size):
        super(reshape, self).__init__()
        self.size = size
        
    def forward(self, x):
        return x.view((x.size()[0],) + self.size)

#the global part of Generator
class Global(nn.Module):
    def __init__(self, channel_num = 3, nz = 100, num_classes=1000):
        super(Global, self).__init__()
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
                    reshape(-1),
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
        self.feat8 = nn.Sequential(
                    nn.Linear(256 + nz, 4096),
                    reshape(64, 8, 8),
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
                    nn.ConvTranspose2d(576, 512, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace = True)
                )
        
        #32 x 32 x 256
        self.deconv1 = nn.Sequential(
                    nn.ConvTranspose2d(768, 256, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace = True)
                )
        
        #64 x 64 x 128
        self.deconv2 = nn.Sequential(
                    nn.ConvTranspose2d(419, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace = True)
                )
        
        #128 x 128 x 64
        self.deconv3 = nn.Sequential(
                    nn.ConvTranspose2d(211, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace = True)
                )
        
        #128 x 128 x 64
        self.conv5 = nn.Sequential(
                    nn.Conv2d(139, 64, kernel_size = 5, stride = 1, padding = 2),
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

    def forward(self, x128, x64, x32, z):
        #Encoder
        conv0 = self.conv0(x128)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        fc = self.fc1(conv4)
        fc = torch.split(fc, 256, 1)
        fc = torch.max(fc[0], fc[1])
        extra = self.extra(fc)

        #Decoder
        #concatenate fc and noise z for the input of self.feat8
        feat8 = self.feat8(torch.cat((fc, z), dim = 1))
        feat32 = self.feat32(feat8)
        feat64 = self.feat64(feat32)
        feat128 = self.feat128(feat64)
        #concatenate feat8 and conv4
        out = self.deconv0(torch.cat((feat8, conv4), dim = 1))
        #concatenate deconv and conv3
        out = self.deconv1(torch.cat((out, conv3), dim = 1))
        #concatenate deconv, feat32, conv2 and x32
        out = self.deconv2(torch.cat((out, feat32, conv2, x32), dim = 1))
        #concatenate deconv, feat64, conv1 and x64
        out = self.deconv3(torch.cat((out, feat64, conv1, x64), dim = 1))
        #concatenate deconv, feat128, conv0, x
        out = self.conv5(torch.cat((out, feat128, conv0, x128), dim = 1))
        out = self.conv6(out)
        out = self.conv7(out)
        return out, extra

#this structure refers to DR-GAN, because the paper don't contains any detail of this network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #128 x 128 x32
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        
        #128 x 128 x 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        
        #64 x 64 x 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        
        #64 x 64 x 64
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        
        #64 x 64 x 128
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ELU()
        )
        
        #32 x 32 x 128
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ELU()
        )
        
        #32 x 32 x 96
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(96),
            nn.ELU()
        )
        
        #32 x 32 x 192
        self.layer8 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(192),
            nn.ELU()
        )
        
        #16 x 16 x 192
        self.layer9 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(192),
            nn.ELU()
        )
        
        #16 x 16 x 128
        self.layer10 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ELU()
        )
        
        #16 x 16 x 256
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ELU()
        )
        
        #8 x 8 x 256
        self.layer12 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ELU()
        )
        
        #8 x 8 x 160
        self.layer13 = nn.Sequential(
            nn.Conv2d(256, 160, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(160),
            nn.ELU()
        )
        
        #8 x 8 x 320
        self.layer14 = nn.Sequential(
            nn.Conv2d(160, 320, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(320),
            nn.ELU()
        )
        
        #AvgPool 1 x 1 x 320
        self.layer15 = nn.Sequential(
            nn.AvgPool2d(kernel_size = 8, stride = 4),
            nn.BatchNorm2d(320),
            nn.ELU()
        )
        
        #1 x 2 x 2
        self.layer16 = nn.Sequential(
            nn.Conv2d(320, 1, kernel_size = 1, stride = 1),
            nn.Sigmoid(),
            reshape(-1),
        )
        
    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        layer7_out = self.layer7(layer6_out)
        layer8_out = self.layer8(layer7_out)
        layer9_out = self.layer9(layer8_out)
        layer10_out = self.layer10(layer9_out)
        layer11_out = self.layer11(layer10_out)
        layer12_out = self.layer12(layer11_out)
        layer13_out = self.layer13(layer12_out)
        layer14_out = self.layer14(layer13_out)
        layer15_out = self.layer15(layer14_out)
        layer16_out = self.layer16(layer15_out)
        return layer16_out
