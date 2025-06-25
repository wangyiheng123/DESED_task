import torch.nn as nn
import torch
import torch.nn.functional as F

class Lt_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u: torch.Tensor,embedings: torch.Tensor):
        u = u.reshape(u.shape[0], 1, u.shape[1])
        embedings = torch.transpose(embedings, 2, 1)
        assert len(u.shape) == 3
        assert len(embedings.shape) == 3

        embeding1 = embedings[:, :, :embedings.shape[2] - 1]

        embedings2 = embedings[:, :, 1:]

        embedings = embeding1 - embedings2  # 直接减

        embedings = torch.transpose(embedings, 1, 2)

        out = F.cosine_similarity(u, embedings, dim=-1)
        return torch.sum(F.relu(out))

class embedingsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(496,27)
        # self.linear.total_ops = torch.DoubleTensor([0])

    def forward(self, x: torch.Tensor):
        return self.linear(x)

