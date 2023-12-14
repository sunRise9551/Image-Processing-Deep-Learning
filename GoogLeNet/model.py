import torch.nn as nn
import torch
import torch.nn.functional as F

class GoogLeNet(nn.Module):
    def __int__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__int__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)   # ceiling 天花板
        self.conv2 = BasicConv2d(64, 64, kernel_size=1, stride=2)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxPool2 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxPool3 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 128, 128)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxPool4 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.averagePool = nn.AvgPool2d(7, 1)
        self.averageAdaptivePool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.7, inplace=True)
        self.fc = nn.Linear(1 * 1 * 1024, num_classes)

        if init_weights:
            self._initialize_weights()

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(512, num_classes)

    def forward(self, x):       # input [B, 3, 224, 224]

        x = self.conv1(x)           # output [B, 64, 112, 112]
        x = self.maxPool1(x)        # output [B, 64, 56, 56]
        x = self.conv2(x)           # output [B, 64, 56, 56]
        x = self.conv3(x)           # output [B, 192, 56, 56]
        x = self.maxPool2(x)        # output [B, 192, 28, 28]

        x = self.inception3a(x)     # output [B, 256, 28, 28]
        x = self.inception3b(x)     # output [B, 480, 28, 28]
        x = self.maxPool3(x)        # output [B, 480, 14, 14]
        x = self.inception4a(x)     # output [B, 512, 14, 14]
        if self.training and self.aux_logits:   # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)     # output [B, 512, 14, 14]
        x = self.inception4c(x)     # output [B, 512, 14, 14]
        x = self.inception4d(x)     # output [B, 528, 14, 14]
        if self.training and self.aux_logits:   # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)     # output [B, 832, 14, 14]
        x = self.maxPool4(x)        # output [B, 832, 7, 7]
        x = self.inception5a(x)     # output [B, 832, 7, 7]
        x = self.inception5b(x)     # output [B, 1024, 7, 7]

        x = self.averageAdaptivePool(x)     # output [B, 1024, 1, 1]
        x = torch.flatten(x, start_dim=1)   # output [B, 1024]
        x = self.dropout(x)         # output [B, 1024]
        x = self.fc(x)              # output [B, num_classes]

        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x



class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=1, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)  # [BATCH, CHANNEL, HEIGHT, WIDTH]


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux).__init__()

        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=1)   #output[BATCH, 128, 4, 4]
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.averagePool(x)
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = F.dropout(x, 0.7, inplace=True, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.7, inplace=True, training=self.training)
        x = self.fc2(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
