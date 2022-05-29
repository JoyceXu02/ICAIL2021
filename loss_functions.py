
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The original implmentation is written by Michal Haltuf on Kaggle.

Returns
	-------
	torch.Tensor
		`ndim` == 1. epsilon <= val <= 1

	Reference
	---------
	- https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
	- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
	- https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
	- http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/

'''
class F1Loss(nn.Module):

	def __init__(self, epsilon = 1e-07):
		super().__init__()
		self.epsilon = epsilon

	def forward(self, y_pred, y_true):
		assert y_pred.ndim == 2
		assert y_true.ndim == 1
		y_true = F.one_hot(y_true, 4).to(torch.float32)
		y_pred = F.softmax(y_pred, dim =1)

		tp = (y_true*y_pred).sum(dim=0).to(torch.float32)
		tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
		fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
		fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)


		precision = tp / (tp + fp + self.epsilon)
		recall = tp / (tp + fn + self.epsilon)

		f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
		f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
		return 1 - f1.mean()


# based on https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938a
class FocalLoss(nn.Module):
	def __init__(self, alpha = 0.25, gamma = 2 ):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha

	def forward(self, in_put, target):
		BCE_loss =nn.CrossEntropyLoss()(in_put, target)
		target = target.type(torch.long)
		pt = torch.exp(-BCE_loss)
		F_loss = self.alpha*(1-pt)**self.gamma*BCE_loss

		return F_loss.mean()


