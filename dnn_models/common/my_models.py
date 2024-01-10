import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class CommonNet(nn.Module):
    def __init__(self, label_num: int) -> None:
        super(CommonNet, self).__init__()

        self.features = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).features

        # 最初のCNN層のチャネル数を倍にする(concatした2枚の画像を入力するため)
        first_conv2d = self.features[0][0]
        self.features[0][0] = torch.nn.Conv2d(
                in_channels=first_conv2d.in_channels * 2,
                out_channels=first_conv2d.out_channels,
                kernel_size=first_conv2d.kernel_size,
                stride=first_conv2d.stride,
                padding=first_conv2d.padding,
                bias=first_conv2d.bias)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        out_channels_of_features: int = self.features[-1][0].out_channels
        self.classifier = nn.Sequential(
                    nn.Dropout(0.2, inplace=False),
                    nn.Linear(out_channels_of_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2, inplace=False),
                    nn.Linear(256, label_num)
                )

    def forward(self, input1, input2):
        # concatenate
        x = torch.cat((input1,input2), dim=1)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)

class DirectionNet(CommonNet):
    def __init__(self) -> None:
        super(DirectionNet, self).__init__(label_num=5)

class OrientationNet(CommonNet):
    def __init__(self) -> None:
        super(OrientationNet, self).__init__(label_num=3)

class RelPosNet(CommonNet):
    def __init__(self) -> None:
        super(RelPosNet, self).__init__(label_num=3)

def test():
    tensor = torch.zeros(2, 3, 224, 224)
    direction_net = DirectionNet()
    orientation_net = OrientationNet()
    relpos_net = RelPosNet()

    output = direction_net(tensor, tensor)
    print(f"direction_net output_shape: {output.shape}")
    output = orientation_net(tensor, tensor)
    print(f"orientation_net output_shape: {output.shape}")
    output = relpos_net(tensor, tensor)
    print(f"relpos_net output_shape: {output.shape}")

if __name__ == "__main__":
    test()
