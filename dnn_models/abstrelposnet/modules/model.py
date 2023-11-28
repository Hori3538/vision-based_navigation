import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn.functional as F

class AbstRelPosNet(nn.Module):
    def __init__(self, bin_num: int = 3) -> None:
        super(AbstRelPosNet, self).__init__()

        efficient = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # efficient = efficientnet_b0(pretrained=True)
        # efficient.classifier = nn.Identity()
        efficient.classifier = nn.Sequential(nn.Identity())

        self.efficient = efficient
        self.fc_layers = nn.Sequential(
                    nn.Linear(1280 * 2, 256),
                    nn.ReLU(),
                    nn.Linear(256, bin_num*2 + 1)
                )

    def forward(self, input1, input2):
        x1 = self.efficient(input1)
        x2 = self.efficient(input2)
        x = torch.cat((x1, x2), dim=1)

        return self.fc_layers(x)

def test():
    tensor = torch.zeros(2, 3, 224, 224)
    model = AbstRelPosNet()
    output = model(tensor, tensor)
    print(output.shape)
    print(output)

if __name__ == "__main__":
    test()
