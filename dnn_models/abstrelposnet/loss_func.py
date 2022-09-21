import torch
from torch import nn

class AbstPoseLoss(nn.Module):
    def __init__(self, beta, device) -> None:
        super(AbstPoseLoss, self).__init__()

        self._weight = torch.tensor([1.0, 1.0, beta], device=device)

    def forward(self, outputs, targets):
        return (self._weight * (outputs - targets) ** 2).mean()

def test():
    outputs = torch.tensor([
            [0, 0, 1],
            [1, 1, 0.7]
        ], dtype=torch.float32)
    targets = torch.tensor([
            [0, 0, 3],
            [3, 3, 0.7]
        ], dtype=torch.float32)

    criterion = AbstPoseLoss(2.0, "cpu")
    loss = criterion(outputs, targets)
    print(loss)

if __name__ == "__main__":
    test()
