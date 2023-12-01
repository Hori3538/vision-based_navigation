import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn.functional as F

class AbstRelPosNet(nn.Module):
    def __init__(self, bin_num: int = 3) -> None:
        super(AbstRelPosNet, self).__init__()

        self.features = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        out_channels_of_features: int = self.features[-1][0].out_channels
        # print(f"out_channels: {out_channels_of_features}")
        self.classifier = nn.Sequential(
                    nn.Linear(out_channels_of_features * 2, 256),
                    nn.ReLU(),
                    nn.Linear(256, bin_num*2 + 1)
                )
        # efficient = efficientnet_b0(weights=None)
        # efficient = efficientnet_b0(pretrained=True)
        # efficient.classifier = nn.Identity()

        # features.classifier = nn.Sequential(nn.Identity())

        # self.efficient = features
        # print(efficient.features[-1].shape)
        # self.fc_layers = nn.Sequential(
        #             nn.Linear(1280 * 2, 256),
        #             nn.ReLU(),
        #             nn.Linear(256, bin_num*2 + 1)
        #         )

    def forward(self, input1, input2):
        # input_concatenated = torch.cat((input1,input2), dim=1)
        # x = self.efficient(input_concatenated)
        # print(f"cccat: {torch.cat((input1,input2), dim=1).shape}")
        x1 = self.features(input1)
        # print(f"x1 shape: {x1.shape}")
        x2 = self.features(input2)
        # print(f"x2 shape: {x2.shape}")
        x = torch.cat((x1, x2), dim=1)
        # print(f"x shape: {x.shape}")
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)

def test():
    tensor = torch.zeros(2, 3, 224, 224)
    model = AbstRelPosNet()
    # first_conv2d = model.efficient.features[0][0]
    # model.efficient.features[0][0] = torch.nn.Conv2d(
    #         in_channels=first_conv2d.in_channels * 2,
    #         out_channels=first_conv2d.out_channels,
    #         kernel_size=first_conv2d.kernel_size,
    #         stride=first_conv2d.stride,
    #         padding=first_conv2d.padding,
    #         bias=first_conv2d.bias)
    # model.efficient.features[0][0].in_channels = 6
    output = model(tensor, tensor)
    print(output.shape)
    # print(output)
    # print(model.efficient.features[0][0].in_channels)
    # print(model.efficient.features[-1][0].out_channels)
    # model.efficient.features[0][0] = torch.nn.Conv2d(6, 32, kernel_size=3, stride=2,
    #                                                 padding=1, bias=False)

if __name__ == "__main__":
    test()
