from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['MobileNetV2']

model_urls = {
    'mobilenet_v2':
    'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _get_deconv_cfg(deconv_kernel):
    if deconv_kernel == 4:
        padding = 1
        output_padding = 0
    elif deconv_kernel == 3:
        padding = 1
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 0
        output_padding = 0

    return deconv_kernel, padding, output_padding


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim,
                       hidden_dim,
                       stride=stride,
                       groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Interpolate(nn.Module):
    def __init__(self, scale, mode='nearest'):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode)
        return x


class MobileNetV2(nn.Module):
    def __init__(self,
                 heads,
                 head_conv=0,
                 last_channel=1280,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 use_deconv=True,
                 use_depthwise=False):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        self.inplanes = last_channel
        self.deconv_with_bias = False
        self.heads = heads

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(
                inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(
                                 inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult,
                                        round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel,
                          output_channel,
                          stride,
                          expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(
            ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        if use_deconv:
            self.upsample_layers = self._make_deconv_layer([256, 256, 256],
                                                           [4, 4, 4])
        else:
            self.upsample_layers = self._make_conv_upsample_layer(
                [256, 256, 256], [3, 3, 3], use_depthwise=use_depthwise)

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                layers = []
                if not use_depthwise:
                    layers.append(
                        nn.Conv2d(256,
                                  head_conv,
                                  kernel_size=3,
                                  padding=1,
                                  bias=True)
                    )
                else:
                    layers.extend([
                        nn.Conv2d(256,
                                  256,
                                  kernel_size=3,
                                  padding=1,
                                  groups=256,
                                  bias=False),
                        nn.Conv2d(256, head_conv, 1, bias=True)
                    ])
                layers.extend([
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv,
                              num_output,
                              kernel_size=1,
                              stride=1,
                              padding=0)])
                fc = nn.Sequential(*layers)
            else:
                fc = nn.Conv2d(in_channels=256,
                               out_channels=num_output,
                               kernel_size=1,
                               stride=1,
                               padding=0)
            self.__setattr__(head, fc)

    def _make_deconv_layer(self, num_filters, kernels):
        layers = []
        for planes, kernel_size in zip(num_filters, kernels):
            kernel, padding, output_padding = _get_deconv_cfg(kernel_size)

            layers.append(
                nn.ConvTranspose2d(in_channels=self.inplanes,
                                   out_channels=planes,
                                   kernel_size=kernel,
                                   stride=2,
                                   padding=padding,
                                   output_padding=output_padding,
                                   bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _make_conv_upsample_layer(self,
                                  num_filters,
                                  kernels,
                                  use_depthwise=False):
        layers = []
        for out_planes, kernel_size in zip(num_filters, kernels):
            layers.append(Interpolate(2))
            padding = (kernel_size - 1) // 2
            if not use_depthwise:
                layers.append(
                    nn.Conv2d(self.inplanes,
                              out_planes,
                              kernel_size,
                              padding=padding,
                              bias=False))
            else:
                layers.extend([
                    nn.Conv2d(self.inplanes,
                              self.inplanes,
                              kernel_size,
                              padding=padding,
                              groups=self.inplanes,
                              bias=False),
                    nn.Conv2d(self.inplanes, out_planes, 1, bias=False)
                ])
            layers.append(nn.BatchNorm2d(out_planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = out_planes

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.upsample_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for head in self.heads:
            final_layer = self.__getattr__(head)
            for m in final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    if m.weight.shape[0] == self.heads[head]:
                        if 'hm' in head:
                            nn.init.constant_(m.bias, -2.19)
                        else:
                            nn.init.normal_(m.weight, std=0.001)
                            nn.init.constant_(m.bias, 0)
        url = model_urls['mobilenet_v2']
        pretrained_state_dict = model_zoo.load_url(url)
        print('=> loading pretrained model {}'.format(url))

        if self.last_channel != 1280:
            keys_to_remove = list(
                filter(lambda k: k.startswith('features.18.'),
                       pretrained_state_dict.keys()))
            for key in keys_to_remove:
                pretrained_state_dict.pop(key)

        self.load_state_dict(pretrained_state_dict, strict=False)

    def forward(self, x):
        x = self.features(x)

        x = self.upsample_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)

        return [ret]


def get_mobilenet_v2(heads, head_conv, **kwargs):
    model = MobileNetV2(heads,
                        head_conv=head_conv,
                        last_channel=256,
                        use_deconv=False,
                        use_depthwise=True)
    model.init_weights()
    return model
