import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn.functional as F
from onehot_conversion import create_onehot_from_output

class AbstRelPosNet(nn.Module):
    def __init__(self, bin_num: int = 3) -> None:
        super(AbstRelPosNet, self).__init__()

        efficient1 = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # efficient1 = efficientnet_b0(pretrained=True)
        # efficient1.classifier = nn.Identity()
        efficient1.classifier = nn.Sequential(nn.Identity())

        efficient2 = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # efficient2 = efficientnet_b0(pretrained=True)
        # efficient2.classifier = nn.Identity()
        efficient2.classifier = nn.Sequential(nn.Identity())

        self.efficient1 = efficient1
        self.efficient2 = efficient2
        self.fc_layers = nn.Sequential(
                    nn.Linear(1280 * 2, 256),
                    nn.ReLU(),
                    nn.Linear(256, bin_num*2 + 1)
                )

    def forward(self, input1, input2):
        x1 = self.efficient1(input1)
        x2 = self.efficient2(input2)
        x = torch.cat((x1, x2), dim=1)

        return self.fc_layers(x)

def test():
    tensor = torch.zeros(2, 3, 224, 224)
    model = AbstRelPosNet()
    output = model(tensor, tensor)
    print(output.shape)
    print(output)
    print(output[:, :4])
    print(F.one_hot(output[:, :4].max(1).indices, num_classes=4))
    one_hot = F.one_hot(output[:, :4].max(1).indices, num_classes=4)
    print(one_hot == one_hot)
    print(one_hot == 1)
    print((one_hot == one_hot) == (one_hot == 1))
    print(torch.sum((one_hot == one_hot) == (one_hot == 1), 0))
    # print(create_onehot_from_output(output))

if __name__ == "__main__":
    test()
