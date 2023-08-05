# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 11:41:28 2020

@author: alfah
"""
import torch
import torch.nn as nn
#import eisen
#from eisen.utils import ModelParallel


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=(3,3,3), padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.conv2 = nn.Conv3d(mid_ch, out_ch, kernel_size=(3,3,3), padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class Nested_UNet(nn.Module):

    def __init__(self, in_ch=1, out_ch=1):
        super(Nested_UNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)).to('cuda:3')
        self.Up = nn.Upsample(scale_factor=(1,2,2)).to('cuda:3')

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0]).to('cuda:0')
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1]).to('cuda:1')
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2]).to('cuda:2')
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3]).to('cuda:3')
        #self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4]).to('cuda:3')

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0]).to('cuda:1')
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1]).to('cuda:2')
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2]).to('cuda:3')
       # self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3]).to('cuda:3')

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0]).to('cuda:0')#
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1]).to('cuda:2')
        #self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2]).to('cuda:2')

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0]).to('cuda:3')
        #self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1]).to('cuda:1')

        #self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0]).to('cuda:3')

        self.final = nn.Conv3d(filters[0], out_ch, kernel_size=1).to('cuda:0')
        self.sigmoid = nn.Sigmoid().to('cuda:0')

    def forward(self, x):
        
        x0_0 = self.conv0_0(x.to('cuda:0')) # 0 - 0
        x1_0 = self.conv1_0(self.pool(x0_0.to('cuda:3')).to('cuda:1')) # 0 - 1
        x0_1 = self.conv0_1(torch.cat([x0_0.to('cuda:1'), self.Up(x1_0.to('cuda:3')).to('cuda:1')], 1))  # 0, 1 - 1

        x2_0 = self.conv2_0(self.pool(x1_0.to('cuda:3')).to('cuda:2')) # 1 - 2
        x1_1 = self.conv1_1(torch.cat([x1_0.to('cuda:2'), self.Up(x2_0.to('cuda:3')).to('cuda:2')], 1))  # 1, 2 - 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1.to('cuda:0'), self.Up(x1_1.to('cuda:3')).to('cuda:0')], 1))  # 0, 1, 2 - 0

        x3_0 = self.conv3_0(self.pool(x2_0.to('cuda:3')).to('cuda:3')) # 2 - 3
        x2_1 = self.conv2_1(torch.cat([x2_0.to('cuda:3'), self.Up(x3_0)], 1))  # 2, 3 - 3
        x1_2 = self.conv1_2(torch.cat([x1_0.to('cuda:2'), x1_1, self.Up(x2_1).to('cuda:2')], 1)) # 1, 2, 3 - 2
        x0_3 = self.conv0_3(torch.cat([x0_0.to('cuda:3'), x0_1.to('cuda:3'), x0_2.to('cuda:3'), self.Up(x1_2.to('cuda:3'))], 1))  # 0, 1, 1, 2 - 3
        """
        x0_0 = self.conv0_0(x) # 0 - 0
        x1_0 = self.conv1_0(self.pool(x0_0)) # 0 - 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))  # 0, 1 - 1

        x2_0 = self.conv2_0(self.pool(x1_0)) # 1 - 2
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))  # 1, 2 - 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))  # 0, 1, 2 - 0

        x3_0 = self.conv3_0(self.pool(x2_0)) # 2 - 3
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))  # 2, 3 - 3
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1)) # 1, 2, 3 - 2
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))
        """
        """
        x4_0 = self.conv4_0(self.pool(x3_0)) 
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1)) 
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))  
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))                   
        """
        output = self.final(x0_3.to('cuda:0')) # 3 - 0
        sigmoidoutput = self.sigmoid(output)
        return sigmoidoutput


device = torch.device('cuda:0')
model = Nested_UNet()

#model = ModelParallel(model,
#                      split_size = 4,
#                      device_ids = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
#                     output_device = 'cuda:0').cuda()
for i in range(33, 50):
    x = torch.randn(1,1,42, 512, 512) #.to(device)
    x = torch.tensor(x, dtype=torch.float32, device=device)
    print(x.shape)
    
    v = model(x)   
    print(v.shape)
    print(v.dtype)

