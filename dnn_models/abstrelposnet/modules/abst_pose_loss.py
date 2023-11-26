import torch
import torch.nn as nn
import torch.nn.functional as F
# from balanced_loss import Loss as FocalLoss
from focal_loss import Loss as FocalLoss

class AbstPoseLoss(nn.Module):
    def __init__(self, class_num=3,
            class_balance_beta: float = 0.999, focal_gamma=2, samples_per_class=None,
            class_balanced=False):
        super(AbstPoseLoss, self).__init__()
        self._label_bin_num = class_num
        # self._loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")
        # self._loss_func = torch.nn.BCEWithLogitsLoss(size_average=True, reduce=False)
        self._loss_func = FocalLoss(
                beta=class_balance_beta,
                fl_gamma=focal_gamma,
                samples_per_class=samples_per_class,
                class_balanced=class_balanced)

    def forward(self, outputs, targets):

        # ce_loss = F.cross_entropy(input, target,reduction="none",weight=self.weight)
        # print(ce_loss)
        # pt = torch.exp(-ce_loss)
        # print(pt)
        # focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        # loss_direction_label = self._loss_func(outputs[:, :self._label_bin_num+1],
        #                                        targets[:, :self._label_bin_num+1])
        # loss_orientation_label = self._loss_func(outputs[:, self._label_bin_num+1:],
        #                                          targets[:, self._label_bin_num+1:])
        
        return self._loss_func(outputs, targets)
        # return loss_direction_label + loss_orientation_label
def test():
    # outputs = torch.tensor([
    #         [0.1, 0.2, 0.7, 0.3, 0.4, 0.3, 0.1],
    #         [0.1, 0.2, 0.7, 0.3, 0.4, 0.3, 0.1]
    #     ], dtype=torch.float32)
    # targets = torch.tensor([
    #         [0, 0, 1, 0, 1, 0, 0],
    #         [1, 0, 0, 0, 1, 0, 0]
    #     ], dtype=torch.float32)
    outputs = torch.tensor([
            [0.1, 0.2, 0.7, 0.3],
            [0.1, 0.2, 0.7, 0.3]
        ], dtype=torch.float32)
    targets = torch.tensor([
            [0, 0, 1, 0],
            [1, 0, 0, 0]
        ], dtype=torch.float32)

    # criterion = AbstPoseLoss(focal_gamma=1, samples_per_class=[5, 5, 20, 5], class_balanced=True, class_balance_beta=0.999)
    criterion = FocalLoss(fl_gamma=1, samples_per_class=[5, 5, 5, 15], class_balanced=True, beta=0.99)
    loss = criterion(outputs, targets)
    print(loss)

if __name__ == "__main__":
    test()
