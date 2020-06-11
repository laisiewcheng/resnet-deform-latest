# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:05:29 2020

@author: User
"""


import torch
from torch import nn
import torch.nn.functional as F

import deform_conv

class PlainNet(nn.Module):
    def __init__(self):
        super(PlainNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias = False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False)
        self.bn4 = nn.BatchNorm2d(128)

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        # convs
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        #print("x1: ", x.size())
        x = F.avg_pool2d(x, 28)
        #print("x2: ", x.size())
        x = x.view(x.size(0), -1)
       
        #x = F.avg_pool2d(x, kernel_size=28, stride=1).view(x.size(0), -1)
        #print("x3: ", x.size())
        x = self.classifier(x)
        #print("x4: ", x.size())
        return F.log_softmax(x, dim=1)


class DeformNet(nn.Module):
    def __init__(self):
        super(DeformNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias = False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = deform_conv.DeformConv2D(128, 128, kernel_size=3, padding=1, bias = False)
        self.bn4 = nn.BatchNorm2d(128)

        #self.classifier = nn.Linear(128, 10)
        self.classifier = nn.Linear(128, 10)


    def forward(self, x):
        # convs
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        # deformable convolution
        x = F.relu(self.conv4(x))
        x = self.bn4(x)

#        x = F.avg_pool2d(x, 28)
#        x = x.view(x.size(0), -1)
#        x = self.classifier(x)
#        
#        return F.log_softmax(x, dim=1)
    
        x = F.avg_pool2d(x, 28)
        #print("x2: ", x.size())
        x = x.view(x.size(0), -1)
       
        #x = F.avg_pool2d(x, kernel_size=28, stride=1).view(x.size(0), -1)
        #print("x3: ", x.size())
        x = self.classifier(x)
        #print("x4: ", x.size())
        return F.log_softmax(x, dim=1)



        #x = F.avg_pool2d(x, kernel_size=28, stride=1).view(x.size(0), -1)
        #x = self.classifier(x)

        #return F.log_softmax(x, dim=1)