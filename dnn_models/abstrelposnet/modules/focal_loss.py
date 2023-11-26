import numpy as np
import torch
import torch.nn.functional as F

def focal_loss(output, labels, alpha=None, gamma=2):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    # output scalar->probability
    output_probability = F.softmax(output, dim=1)
    # print(f"prob: {output_probability}")
    # bc_loss = F.binary_cross_entropy_with_logits(input=output, target=labels, reduction="none")
    bc_loss = F.binary_cross_entropy(input=output_probability, target=labels, reduction="none")
    # bc_loss = F.cross_entropy(input=output_probability, target=labels, reduction="none")
    # print(f"bc_loss: {bc_loss}")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * output_probability - gamma * torch.log(1 + torch.exp(-1.0 * output_probability)))

    print(f"modulator: {modulator}")
    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        print(f"alpha: {alpha}")
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= len(labels) # mean of batch
    # focal_loss /= torch.sum(labels) # mean of batch
    return focal_loss

class Loss(torch.nn.Module):
    def __init__(
        self,
        beta: float = 0.999,
        fl_gamma=2,
        samples_per_class=None,
        class_balanced=False,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf

        Args:
            beta: float. Hyperparameter for Class balanced loss.
            fl_gamma: float. Hyperparameter for Focal loss.
            samples_per_class: A python list of size [num_classes].
                Required if class_balance is True.
            class_balanced: bool. Whether to use class balanced loss.
        Returns:
            Loss instance
        """
        super(Loss, self).__init__()

        if class_balanced is True and samples_per_class is None:
            raise ValueError("samples_per_class cannot be None when class_balanced is True")

        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced

    def forward(
        self,
        logits: torch.tensor,
        labels: torch.tensor,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """

        batch_size = logits.size(0)
        num_classes = logits.size(1)
        # labels_one_hot = F.one_hot(labels, num_classes).float()

        if self.class_balanced:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * num_classes
            weights = torch.tensor(weights, device=logits.device).float()

            # weights = weights.unsqueeze(0)
            # # weights = weights.repeat(batch_size, 1) * labels_one_hot
            # weights = weights.repeat(batch_size, 1) * labels
            # weights = weights.sum(1)
            # weights = weights.unsqueeze(1)
            # weights = weights.repeat(1, num_classes)
        else:
            weights = None

        # cb_loss = focal_loss(logits, labels_one_hot, alpha=weights, gamma=self.fl_gamma)
        cb_loss = focal_loss(logits, labels, alpha=weights, gamma=self.fl_gamma)

        return cb_loss
