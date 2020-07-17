import torch.nn as nn

def _make_layers(cfg, batch_norm=False, batch_norm_params=None, in_channels = 3):
    layers = []
    layer_map = {}
    i1 = 1
    i2 = 1
    if batch_norm_params is None:
        batch_norm_params = {'eps':1e-05, 'momentum':0.1}
    for v in cfg:
        if v == 'M':
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            layers += [pool]
            layer_map['pool%d'%i1] = pool
            i1 += 1
            i2 = 1
        else:
            stride = 1
            if isinstance(v, tuple):
                stride = v[1]
                v = v[0]
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=stride, padding=1)
            layer_map['conv%d_%d'%(i1,i2)] = conv2d
            i2 += 1
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, batch_norm_params['eps'], batch_norm_params['momentum']), LossBuilder._lrelu()]
            else:
                layers += [conv2d, nn.LeakyReLU()]
            in_channels = v
    return nn.Sequential(*layers), layer_map
