import torch
import joblib
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

class em_model():
    def __init__(self, dense_weight = 0.0 ,sparse_weight = 0.0, learning_rate = 1e-5):
        self.sparse_weight = torch.tensor([sparse_weight], requires_grad=True)
        self.dense_weight = torch.tensor([dense_weight], requires_grad=True)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam([self.dense_weight, self.sparse_weight], lr= self.learning_rate)

    def forward(self, x):
        dense_scores , sparse_scores = x
        return self.dense_weight*dense_scores+ self.sparse_weight * sparse_scores

    def loss(self, score, target):
        """
        sum all scores among positive samples
        """
        predict = F.softmax(score, dim=-1)
        loss = predict * target
        loss = loss.sum(dim=-1)                   # sum all positive scores
        loss = loss[loss > 0]                     # filter sets with at least one positives
        loss = torch.clamp(loss, min=1e-9, max=1) # for numerical stability
        loss = -torch.log(loss)                   # for negative log likelihood
        if len(loss) == 0:
            loss = loss.sum()                     # will return zero loss
        else:
            loss = loss.mean()
        return loss

    def __param__(self):
        return self.dense_weight, self.sparse_weight

    def __call__(self,x):
        scores = self.forward(x)
        return scores