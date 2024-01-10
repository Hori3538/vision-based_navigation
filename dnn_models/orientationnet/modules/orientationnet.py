import torch

from my_models import CommonNet

class OrientationNet(CommonNet):
    def __init__(self) -> None:
        super(OrientationNet, self).__init__(label_num=3)

def test():
    tensor = torch.zeros(2, 3, 224, 224)
    model = OrientationNet()
    output = model(tensor, tensor)
    print(output.shape)

if __name__ == "__main__":
    test()
