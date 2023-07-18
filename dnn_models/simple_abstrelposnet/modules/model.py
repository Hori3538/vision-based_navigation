import torch
import torch.nn as nn
import torchvision
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SimpleAbstRelPosNet(nn.Module):
    def __init__(self) -> None:
        super(SimpleAbstRelPosNet, self).__init__()

        efficient1 = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        efficient1.features[-1] = torchvision.ops.misc.Conv2dNormActivation(320, 100, kernel_size=(1,1), activation_layer=nn.SiLU)
        # efficient1 = efficientnet_b0(pretrained=True)
        efficient1.classifier = nn.Sequential(nn.Identity())

        efficient2 = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        efficient2.features[-1] = torchvision.ops.misc.Conv2dNormActivation(320, 100, kernel_size=(1,1), activation_layer=nn.SiLU)
        # efficient2 = efficientnet_b0(pretrained=True)
        efficient2.classifier = nn.Sequential(nn.Identity())

        self.efficient1 = efficient1
        self.efficient2 = efficient2
        self.fc_layers = nn.Sequential(
                    nn.Linear(100 * 2, 32),
                    nn.ReLU(),
                    nn.Linear(32, 4)
                )

    def forward(self, input1, input2):
        x1 = self.efficient1(input1)
        x2 = self.efficient2(input2)
        x = torch.cat((x1, x2), dim=1)

        return self.fc_layers(x)

def test():
    tensor = torch.zeros(1, 3, 224, 224)
    model = SimpleAbstRelPosNet()
    output = model(tensor, tensor)
    print(output.shape)
    print(output)

if __name__ == "__main__":
    test()
