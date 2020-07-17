import torch
import torch.nn as nn
import torch.nn.functional as F

#"Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al."
# https://github.com/pytorch/examples/tree/master/super_resolution
class SubpixelNet(nn.Module):
    def __init__(self, upscale_factor, input_channels, channel_mask, output_channels, opt):
        super(SubpixelNet, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channels, 64, (5, 5), (1, 1), (2, 2)) # Color input + Mask
        self.conv2 = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(32, output_channels * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pixel_shuffle(self.conv5(x))
        return x, None

    def _initialize_weights(self):
        torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
        torch.nn.init.orthogonal_(self.conv2.weight, torch.nn.init.calculate_gain('relu'))
        torch.nn.init.orthogonal_(self.conv3.weight, torch.nn.init.calculate_gain('relu'))
        torch.nn.init.orthogonal_(self.conv4.weight, torch.nn.init.calculate_gain('relu'))
        torch.nn.init.orthogonal_(self.conv5.weight)