# Based on "DeepFovea - Neural Reconstruction for Foveated Rendering and Video Compression using Learned Statistics of Natural Videos"

import torch
from torch import nn
import torch.nn.functional as F

class DeepFoveaNet(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        *,
        depth=5,
        features=[32, 64, 128, 128, 128],
        activation='elu',
        pool='avg',
        residual=False
    ):
        """
        Implementation of DeepFovea

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            depth (int): depth of the network
            features: list of length 'depth' with the number of channels in the inner layers
            activation: activation function (case-insensitive), 
                can be 'relu', 'elu' (default), 'leakyrelu'
            pool: pooling layer (case-insensitive), can be 'max', 'avg' (default)
            residual: add residual connections also from the input to the output,
                not only in the inner blocks
        """
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._depth = depth
        assert len(features)==depth, "length of feature list does not match depth"
        self._features = features
        assert activation.lower() in ["relu", "elu", "leakyrelu"], "unknown activation "+activation
        self._activation = activation.lower()
        assert pool.lower() in ["avg", "max"], "unknown pooling layer "+pool
        self._pool = pool.lower()
        self._residual = residual

        # build down-path
        prev_channels = in_channels
        self._down_blocks = nn.ModuleList()
        self._down_pool = nn.ModuleList()
        for i in range(depth):
            c = features[i]
            if i==0:
                self._down_pool.append(self._feature_change_block(prev_channels, c))
            else:
                self._down_pool.append(nn.Sequential(
                    DeepFoveaNet._get_pool(self._pool),
                    self._feature_change_block(prev_channels, c)
                    ))
            self._down_blocks.append(self._residual_block(c))
            prev_channels = c

        # build up-path
        self._up_blocks = nn.ModuleList()
        self._up_pool = nn.ModuleList()
        for i in range(depth-2, -1, -1):
            c = features[i]
            self._up_pool.append(nn.Sequential(
                self._feature_change_block(prev_channels, c),
                self._get_unpool()
                ))
            self._up_blocks.append(self._temporal_block(c))
            prev_channels = c
        self._last_block = self._feature_change_block(prev_channels, out_channels)

    def forward(self, input, *temp_connection):
        # get shape
        B, C, H, W = input.shape
        # validate temp connection shapes
        temp_connection_shapes = self.get_temporal_shapes(input.shape)
        assert len(temp_connection_shapes)==len(temp_connection)
        for i in range(len(temp_connection_shapes)):
            assert temp_connection_shapes[i] == temp_connection[i].shape

        bridges = []
        temp_out = []

        # downward
        x = input
        for pool, down in zip(self._down_pool, self._down_blocks):
            x = down(pool(x))
            bridges.append(x)

        # upward
        for i,(unpool,up) in enumerate(zip(self._up_pool, self._up_blocks)):
            x, temp = up(unpool(x), bridges[-i-2], temp_connection[i])
            temp_out.append(temp)

        #convolution to the desired output channels
        x = self._last_block(x)

        # residual connections from input to output
        if self._residual:
            if self.in_channels < self.out_channels:
                x = torch.cat([
                    x[:,0:self.in_channels,:,:] + input,
                    x[:,self.in_channels:,:,:]],
                    dim=1)
            elif self.in_channels > self.out_channels:
                x = x + input[:,0:self.out_channels,:,:]
            else:
                x = x + input

        return (x,) + tuple(temp_out)

    @staticmethod
    def _get_activation(activation):
        if activation=='relu':
            return nn.ReLU()
        elif activation=='elu':
            return nn.ELU()
        elif activation=='leakyrelu':
            return nn.LeakyReLU()

    @staticmethod
    def _get_pool(pool):
        if pool=='avg':
            return nn.AvgPool2d(2)
        elif pool=='max':
            return nn.MaxPool2d(2)

    @staticmethod
    def _get_unpool():
        return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def _feature_change_block(self, in_size, out_size):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
            DeepFoveaNet._get_activation(self._activation)
            )

    class ResidualBlock(nn.Module):
        """
        The paper does not mention how they change the feature count so I 
        insert another convolution at the end (after the pooling)
        """
        def __init__(self, features, activation):
            super().__init__()
            self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
            self.activ1 = DeepFoveaNet._get_activation(activation)
            self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
            self.activ2 = DeepFoveaNet._get_activation(activation)

        def forward(self, x):
            y = self.activ2(self.conv2(self.activ1(self.conv1(x))))
            return y + x

    def _residual_block(self, features):
        return DeepFoveaNet.ResidualBlock(features, self._activation)

    class TemporalBlock(nn.Module):
        """
        The temporal bridge has the same shape as the output
        """
        def __init__(self, features, activation):
            super().__init__()
            self.conv1 = nn.Conv2d(3*features, features, kernel_size=3, stride=1, padding=1)
            self.activ1 = DeepFoveaNet._get_activation(activation)
            self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
            self.norm = nn.BatchNorm2d(features)
            self.activ2 = DeepFoveaNet._get_activation(activation)
            self.conv3 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
            self.activ3 = DeepFoveaNet._get_activation(activation)

        def forward(self, lower, bridge, temporal):
            x = torch.cat([lower, bridge, temporal], 1)
            x = self.activ1(self.conv1(x))
            y = self.activ2(self.norm(self.conv2(x)))
            temporal = y
            y = self.activ3(self.conv3(y))
            y = y + x
            return y, temporal

    def _temporal_block(self, features):
        return DeepFoveaNet.TemporalBlock(features, self._activation)

    def get_temporal_shapes(self, input_shape):
        """
        Returns the shape of the temporal connections inside the network
        as a list of tuples.
        These are the shapes that are passed as input 'temp_connection' to 
        the forward method and are also returned for the next iteration.

        Args:
         - input_shape: The expected shape (B, C, H, W) of the input to the network as a tuple
        """
        B, C, H, W = input_shape
        assert C == self._in_channels

        # first downscale
        
        # get feature size (no inner-most layer)
        sizes = []
        for f in self._features[:-1]:
            sizes.append((B, f, H, W))
            H = H // 2
            W = W // 2
        # connections are used in reverse order
        sizes.reverse()
        return sizes

if __name__ == "__main__":
    print("Test DeepFovea network to produce correct shaped outputs")

    batches = 4
    shape = (128,128)

    in_channels = 5
    out_channels = 6
    depth = 5
    features = [32, 64, 128, 128, 128]
    residual = False

    net = DeepFoveaNet(
        in_channels, out_channels,
        depth=depth, features=features,
        residual=residual)
    print("Network:")
    print(net)

    input_shape = (batches, in_channels, shape[0], shape[1])
    temp_shapes = net.get_temporal_shapes(input_shape)    
    print("temporal shapes:", temp_shapes)

    try:
        from torchsummary import summary
        summary(net, 
            input_size=
                [(in_channels, shape[0], shape[1])] + 
                [tuple(list(s)[1:]) for s in net.get_temporal_shapes(input_shape)], 
            device='cpu',
            batch_size = batches)
    except ModuleNotFoundError:
        print("No summary writer found")

    input = torch.rand(*input_shape, dtype=torch.float32)
    temp1 = [torch.rand(*s, dtype=torch.float32) for s in temp_shapes]

    print()
    print("Run input 1")
    output1, *temp2 = net(input, *temp1)
    print("output shape:", output1.shape)

    print()
    print("Run input 2")
    input2 = output1[:,0:5,:,:]
    output2, *temp2 = net(input2, *temp2)
    print("output shape:", output2.shape)