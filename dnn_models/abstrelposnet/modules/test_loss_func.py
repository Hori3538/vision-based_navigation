import torch
from torch import nn
import torch.nn.functional as F

class AbstPoseLoss(nn.Module):
    def __init__(self) -> None:
        super(AbstPoseLoss, self).__init__()

    def forward(self, outputs, targets):
        # loss_x = F.cross_entropy(outputs[:, :3], targets[:, :3], reduction="none")
        loss_x = F.cross_entropy(outputs, targets, reduction="none")
        # loss_x = F.softmax(outputs, dim=1)
        # loss_x = F.log_softmax(outputs, dim=1)
        # print(loss_x)
        # loss_y = F.cross_entropy(outputs[:, 3:6], targets[:, 3:6])
        # loss_z = F.cross_entropy(outputs[:, 6:], targets[:, 6:])

        return loss_x
        # return loss_x + loss_y + loss_z

def test():
    # outputs = torch.tensor([
    #         [0.1, 0.2, 0.7, 0.3, 0.4, 0.3, 0.1, 0.1, 0.8],
    #         [0.1, 0.2, 0.7, 0.3, 0.4, 0.3, 0.1, 0.1, 3.0]
    #     ], dtype=torch.float32)
    # targets = torch.tensor([
    #         [0, 0, 1, 0, 1, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 1, 0, 0, 1, 0]
    #     ], dtype=torch.float32)
    outputs = torch.tensor([
            # [0.1, 0.2, 0.7],
            [0.0, 0.0, 1.0],
            # [0.1, 0.2, 0.7]
            [1.0, 0.0, 0.0]
        ], dtype=torch.float32)
    targets = torch.tensor([
            [0, 0, 1],
            [1, 0, 0]
        ], dtype=torch.float32)

    criterion = AbstPoseLoss()
    loss = criterion(outputs, targets)
    print(loss)

if __name__ == "__main__":
    test()
