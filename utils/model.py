"""We do not use official pytorch ResNet because 
    it is not flexible in inputting image with aspect ratio >> 1
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_channel, last = False):
        super(ConvBlock, self).__init__()
        self.Conv_1 = nn.Conv2d(in_channels=input_channel, out_channels=2*input_channel, kernel_size=(5,5), padding='same')
        self.Conv_2 = nn.Conv2d(in_channels=2*input_channel, out_channels=2*input_channel, kernel_size=(5,5), padding='same')
        self.Conv_1l = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=(5,5), padding='same')
        self.Conv_2l = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=(5,5), padding='same')
        self.Conv_skip_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=2*input_channel, kernel_size=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*input_channel)
        )
        self.Conv_skip_1l = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(input_channel)
        )
        self.features = nn.Sequential(
            self.Conv_1,
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*input_channel),
            self.Conv_2,
            nn.BatchNorm2d(2*input_channel)
        )
        if last:
            self.features = nn.Sequential(
                self.Conv_1l,
                nn.LeakyReLU(),
                nn.BatchNorm2d(input_channel),
                self.Conv_2l,
                nn.BatchNorm2d(input_channel)
            )
        self.last = last
        self.input_channel = input_channel
        
    def forward(self, input):
        if not self.last:
            input_skip = self.Conv_skip_1(input)
        else:
            input_skip = self.Conv_skip_1l(input)
        input = self.features(input)
        input = input_skip + input
        return nn.ReLU()(input)

class DeConvBlock(nn.Module):
    def __init__(self, input_channel, last = False):
        super(DeConvBlock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=input_channel//2, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(input_channel//2),
            nn.Conv2d(in_channels=input_channel//2, out_channels=input_channel//2, kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(input_channel//2)
        )
        if last: 
            self.features = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=input_channel//4, kernel_size=(3,3), padding='same'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(input_channel//4),
                nn.Conv2d(in_channels=input_channel//4, out_channels=input_channel//4, kernel_size=(3,3), padding='same'),
                nn.BatchNorm2d(input_channel//4)
            )
        self.last = last
        self.input_channel = input_channel

    def forward(self, input):
        if not self.last:
            input_skip = nn.Conv2d(in_channels=self.input_channel, out_channels=self.input_channel//2, kernel_size=(1,1))(input).to(self.device)
            input_skip = nn.LeakyReLU()(input_skip)
            input_skip = nn.BatchNorm2d(self.input_channel//2)(input_skip)
        else:
            input_skip = nn.Conv2d(in_channels=self.input_channel, out_channels=self.input_channel//4, kernel_size=(1,1))(input).to(self.device)
            input_skip = nn.LeakyReLU()(input_skip)
            input_skip = nn.BatchNorm2d(self.input_channel//4)(input_skip)
        input = self.features(input)
        input = input_skip + input
        return nn.ReLU()(input)
    
class ResNet10(nn.Module):
    def __init__(self, input_size = (80, 400), channels=64, out_channels=3, dropout=0.3) -> None:
        super(ResNet10, self).__init__()
        self.ConvBlock_1 = ConvBlock(input_channel=channels, last=True)
        self.ConvBlock_2 = ConvBlock(input_channel=channels)
        self.ConvBlock_3 = ConvBlock(input_channel=2*channels, last=True)
        self.ConvBlock_4 = ConvBlock(input_channel=2*channels)
        self.features_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_1,
        )
        self.features_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_2,
        )
        self.features_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_3,
        )
        self.features_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_4,
        )
        self.decoder = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)), # 1/3 1/4 for EyeDiap
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=int(48*channels), out_features=channels//4),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=channels//4, out_features=out_channels),
        )
        self.channels = channels
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.uniform_(0.0, 1e-3)
            m.bias.data.fill_(0)
            
    def forward(self, input):
        input = self.features_1(input)
        input = self.features_2(input)
        input = self.features_3(input)
        input = self.features_4(input)
        input = self.decoder(input)
        return input
    
    
