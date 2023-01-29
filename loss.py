import torch
import torch.nn as nn

class ClassBCELoss(nn.Module):
    def __init__(self, num_classes=3, weights=[0.4, 0.2, 0.4]) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weights = weights
        self.bce_loss = nn.BCELoss(reduction='mean')
    
    def forward(self, x, y):
        loss = 0 
        for i in range(self.num_classes):
            loss += self.bce_loss(x, y) * self.weights[i]

        return loss


class CELossWithPenalty(nn.Module):
    def __init__(self, weight=5.0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.penalty_weight = weight
        self.trans_matrix = torch.tensor([[0.7, 0.25, 4.5],
                                          [1.5, 0.125, 2.0],
                                          [3.0, 0.3, 0.5]])
    
    def forward(self, input, target):
        pred = torch.softmax(input, dim=1).argmax(dim=1)
        label = target.argmax(dim=1)
        penalty = torch.zeros_like(pred).float()
        for i in range(3):
            for j in range(3):
                penalty[torch.logical_and(label==i, pred==j)] = self.trans_matrix[i, j]

        loss_ce = self.loss(input, target)
        penalty_loss = loss_ce * penalty

        return penalty_loss.mean()


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


