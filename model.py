import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1  # #kernels if change

    # downSample对应虚线的残差结构，需要降维，1x1 conv
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        shortCut = x
        if self.downsample is not None:
            shortCut = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += shortCut
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        ## stride may change for different type of block, so use a input parameter
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        shortCut = x
        if self.downsample is not None:  # dot-dash short cut
            shortCut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortCut
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, block_nums, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channels = 64  # #channels before res block
        # in_channels = 3 (rgb), padding=3 make width height half
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_nums[0])
        self.layer2 = self._make_layer(block, 128, block_nums[1], 2)
        self.layer3 = self._make_layer(block, 256, block_nums[2], 2)
        self.layer4 = self._make_layer(block, 512, block_nums[3], 2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # #feats x #channels x 1 x 1
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # layer intialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.include_top:
            out = self.avgpool(out)
            out = torch.flatten(out, 1)  # (N, C, H, W) -> (N, C*H*W)
            out = self.fc(out)
        return out

    # in_channels, in_channels of first layer of block
    def _make_layer(self, block, in_channels, block_num, stride=1):

        ### example
        ### 18-layer resnet
        ## layer conv2_x 2x block with size 56x56x64, out 56x56x64, s=1
        # blk1 in 56x56x64 out 56x56x64
        # blk2 in 56x56x64 out 56x56x64
        ## layer conv3_x 2x block with in 56x56x64, out 28x28x128, s=2
        # blk1 in 56x56x64 out 28x28x128
        # blk2 in 28x28x128 out 28x28x128

        downsample = None
        # maxpool -> conv2_x s=1, conv2_x -> conv3_x : s=2...
        if stride != 1 or self.in_channels != in_channels * block.expansion:
            downsample = nn.Sequential(
                # downsample needs to shrink size to half and double channels to make the shortCut shape same as output
                nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(in_channels * block.expansion)
            )
        # layers with block
        layers = []
        # self.in_channels: #channels enter block
        # in_channels: #channels of first layer of block
        layers.append(block(self.in_channels, in_channels, downsample=downsample, stride=stride))
        self.in_channels = in_channels * block.expansion

        # for the rest of block in the layer, self.in_channels is updated above, so matched.
        for _ in range(1, block_num):
            layers.append(block(self.in_channels, in_channels))

        return nn.Sequential(*layers)


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, include_top)
