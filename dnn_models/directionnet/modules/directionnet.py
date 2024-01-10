import torch

from my_models import CommonNet

class DirectionNet(CommonNet):
    def __init__(self) -> None:
        super(DirectionNet, self).__init__(label_num=5)

def test():
    tensor = torch.zeros(2, 3, 224, 224)
    model = DirectionNet()
    output = model(tensor, tensor)
    print(output.shape)

if __name__ == "__main__":
    test()
