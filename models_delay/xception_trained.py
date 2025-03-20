import torch
import torch.nn as nn
import pandas as pd
import utils_mine.DelayExpansion as DEX

class DelayExpansionConv2d(nn.Conv2d):
    def __init__(self, delay_layer, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DelayExpansionConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.delay_layer = delay_layer

    def forward(self, input):
        output = super(DelayExpansionConv2d, self).forward(input)
        batch_size = output.size(0)
        # Assuming delay_layer's forward method can handle the output of Conv2d
        self.delay_layer(output, batch_size, self)
        return output

class DelayExpansionLinear(nn.Linear):
    def __init__(self, delay_layer, in_features, out_features, bias=True):
        super(DelayExpansionLinear, self).__init__(in_features, out_features, bias)
        self.delay_layer = delay_layer

    def forward(self, input):
        output = super(DelayExpansionLinear, self).forward(input)
        batch_size = output.size(0)
        assert output.dim() == 2
        self.delay_layer(output, batch_size, self)
        return output


#
data = [
    [0.0, 0.056598642],
    [0.0666667, 0.205962435],
    [0.1333333, 0.312138982],
    [0.2, 0.437158198],
    [0.2666667, 0.319973934],
    [0.3333333, 0.450264408],
    [0.4, 0.559485637],
    [0.4666667, 0.694916383],
    [0.5333333, 0.562896787],
    [0.6, 0.709107365],
    [0.6666667, 0.811728286],
    [0.7333333, 0.939352112],
    [0.8, 0.818508719],
    [0.8666667, 0.958645411],
    [0.9333333, 1.072683293],
    [1.0, 1.192973781]
]
delay_data = pd.DataFrame(data, columns=["data", "delay expension"])
delay_layer = DEX.DelayExpansionLayer(delay_data)
#

class SeperableConv2d(nn.Module):

    #***Figure 4. An “extreme” version of our Inception module,
    #with one spatial convolution per output channel of the 1x1
    #convolution."""
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.depthwise = DelayExpansionConv2d(delay_layer, input_channels, input_channels, kernel_size=kernel_size, groups=input_channels, padding=1, bias=False)
        self.pointwise = DelayExpansionConv2d(delay_layer, input_channels, output_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class EntryFlow(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Sequential(
            DelayExpansionConv2d(delay_layer, 3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            DelayExpansionConv2d(delay_layer, 32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3_residual = nn.Sequential(
            SeperableConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.conv3_shortcut = nn.Sequential(
            DelayExpansionConv2d(delay_layer, 64, 128, 1, stride=2),
            nn.BatchNorm2d(128),
        )

        self.conv4_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv4_shortcut = nn.Sequential(
            DelayExpansionConv2d(delay_layer, 128, 256, 1, stride=2),
            nn.BatchNorm2d(256)
        )

        #no downsampling
        self.conv5_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, 1, padding=1)
        )

        #no downsampling
        self.conv5_shortcut = nn.Sequential(
            DelayExpansionConv2d(delay_layer, 256, 728, 1, stride=1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x = residual + shortcut
        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        x = residual + shortcut
        residual = self.conv5_residual(x)
        shortcut = self.conv5_shortcut(x)
        x = residual + shortcut

        return x

class MiddleFLowBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)

        shortcut = self.shortcut(x)

        return shortcut + residual

class MiddleFlow(nn.Module):
    def __init__(self, block):
        super().__init__()

        #"""then through the middle flow which is repeated eight times"""
        self.middel_block = self._make_flow(block, 8)

    def forward(self, x):
        x = self.middel_block(x)
        return x

    def _make_flow(self, block, times):
        flows = []
        for i in range(times):
            flows.append(block())

        return nn.Sequential(*flows)


class ExitFLow(nn.Module):

    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeperableConv2d(728, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.shortcut = nn.Sequential(
            DelayExpansionConv2d(delay_layer, 728, 1024, 1, stride=2),
            nn.BatchNorm2d(1024)
        )

        self.conv = nn.Sequential(
            SeperableConv2d(1024, 1536, 3, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeperableConv2d(1536, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        output = shortcut + residual
        output = self.conv(output)
        output = self.avgpool(output)

        return output

class Xception(nn.Module):

    def __init__(self, block, num_class=100):
        super().__init__()
        self.entry_flow = EntryFlow()
        self.middel_flow = MiddleFlow(block)
        self.exit_flow = ExitFLow()

        self.fc = DelayExpansionLinear(delay_layer, 2048, num_class)

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middel_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def xception():
    return Xception(MiddleFLowBlock)