
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from datasets import load_dataset,Dataset,DatasetDict

import transformers
from transformers import RobertaModel, RobertaTokenizer


class Roberta(nn.Module):
	def __init__(self, pretrained, output_dim= 4, dropout_rate = 0.2):
		super(Roberta, self).__init__()
		self.model = RobertaModel.from_pretrained(pretrained)
		self.second_last = nn.Linear(768, 768)
		self.dropout = nn.Dropout(dropout_rate)
		self.classifier = nn.Linear(768, output_dim)

	def forward(self, input_ids, attention_mask, token_type_ids):
		output = self.model(input_ids = input_ids,
							attention_mask = attention_mask,
							token_type_ids = token_type_ids)
		hidden = output[0]
		pooler = self.second_last(hidden[:, 0])
		pooler = self.dropout(nn.ReLU()(pooler))
		output = self.classifier(pooler)
		return output
