"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class BreathConv2d(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, tmp_in_cores=1, tmp_out_cores=1, PI=32, PO=32, array_size=16, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.in_cores = int(in_channels/PI)
        self.out_cores = int(out_channels/PO)
        self.array_size = array_size
        self.tmp_in_cores = tmp_in_cores
        self.tmp_out_cores = tmp_out_cores

        self.breath = nn.Sequential(
            # SuperCIM1
            nn.Conv2d(in_channels=in_channels, out_channels=self.tmp_in_cores*PI, kernel_size=1, stride=stride, padding=0),
            #nn.BatchNorm2d(self.tmp_in_cores*PI),
            #nn.ReLU(inplace=True),

            #SuperCIM2
            nn.Conv2d(in_channels=self.tmp_in_cores*PI, out_channels=self.tmp_out_cores*PO, kernel_size=kernel_size, stride=stride, padding=padding, groups=self.tmp_out_cores),
            #nn.BatchNorm2d(self.tmp_out_cores*PO),
            #nn.ReLU(inplace=True),

            #SuperCIM3
            nn.Conv2d(in_channels=self.tmp_out_cores*PO, out_channels=out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if self.in_cores * self.tmp_in_cores > self.array_size:
            print(f"The shrink part requires {int(self.in_cores*self.tmp_in_cores/self.array_size)}")
            print(f"The conv part requires {int(self.tmp_in_cores*self.tmp_out_cores*kernel_size*kernel_size/self.array_size)}")
            print(f"The extent part requires {int(self.tmp_out_cores*self.out_cores/self.array_size)}")

    def forward(self,x):

        breath = self.breath(x)

        return breath


class VGG(nn.Module):
    def __init__(self, num_class=100):
        super().__init__()

        self.branchA1 = BreathConv2d(in_channels=3, out_channels=64, tmp_in_cores=1, tmp_out_cores=1, kernel_size=3, stride=1, padding=1)
        self.branchA2 = BreathConv2d(in_channels=64, out_channels=64, tmp_in_cores=1, tmp_out_cores=1, kernel_size=3, stride=1, padding=1)

        self.branchB1 = BreathConv2d(in_channels=64, out_channels=128, tmp_in_cores=1, tmp_out_cores=2, kernel_size=3, stride=1, padding=1)
        self.branchB2 = BreathConv2d(in_channels=128, out_channels=128, tmp_in_cores=1, tmp_out_cores=2, kernel_size=3, stride=1, padding=1)

        self.branchC1 = BreathConv2d(in_channels=128, out_channels=256, tmp_in_cores=1, tmp_out_cores=2, kernel_size=3, stride=1, padding=1)
        self.branchC2 = BreathConv2d(in_channels=256, out_channels=256, tmp_in_cores=1, tmp_out_cores=4, kernel_size=3, stride=1, padding=1)

        self.branchD1 = BreathConv2d(in_channels=256, out_channels=512, tmp_in_cores=1, tmp_out_cores=4, kernel_size=3, stride=1, padding=1)
        self.branchD2 = BreathConv2d(in_channels=512, out_channels=512, tmp_in_cores=1, tmp_out_cores=4, kernel_size=3, stride=1, padding=1)

        self.branchE1 = BreathConv2d(in_channels=512, out_channels=512, tmp_in_cores=1, tmp_out_cores=4, kernel_size=3, stride=1, padding=1)
        self.branchE2 = BreathConv2d(in_channels=512, out_channels=512, tmp_in_cores=1, tmp_out_cores=8, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self,x):

        #1-2
        x = self.branchA1(x)
        x = self.branchA2(x)
        x = self.pool(x)

        #3-4
        x = self.branchB1(x)
        x = self.branchB2(x)
        x = self.pool(x)

        #5-7
        x = self.branchC1(x)
        x = self.branchC2(x)
        x = self.branchC2(x)
        x = self.pool(x)

        #8-10
        x = self.branchD1(x)
        x = self.branchD2(x)
        x = self.branchD2(x)
        x = self.pool(x)

        #11-13
        x = self.branchE1(x)
        x = self.branchE2(x)
        x = self.branchE2(x)
        x = self.pool(x)

        output = x.view(x.size()[0], -1)
        output = self.classifier(output)

        return output


def vgg11_bn():
    print(f"not defined")
    return VGG()

def vgg13_bn():
    print(f"not defined")
    return VGG()

def vgg16_bn():
    return VGG()


