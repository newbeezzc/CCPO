
import torch
from torchvision import models
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import Bottleneck

__all__ = ['ResNet', 'resnet50', 'resnet101']


class ResNet(models.ResNet):
    """ResNets without fully connected layer"""

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self._out_features = self.fc.in_features
        # del self.fc

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.view(-1, self._out_features)
        return x

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features


def _resnet(arch, block, layers, weights, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], weights, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], weights, progress,
                   **kwargs)