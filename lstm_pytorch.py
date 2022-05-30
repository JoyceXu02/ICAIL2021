
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

## reference: https://github.com/bentrevett/pytorch-sentiment-analysis
class LSTM(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_dim, 
				output_dim, padding_idx,dropout=0.5, n_layers=1, bidirectional=True):
		super(LSTM, self).__init__()
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.hidden_dim = hidden_dim
		self.dropout_rate = dropout
		self.bidirectional = bidirectional

		self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
		self.lstm = nn.LSTM(embed_size, hidden_dim, n_layers, bidirectional=bidirectional,
							dropout=dropout, batch_first = True)
		self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)
		self.dropout = nn.Dropout(self.dropout_rate)

	def forward(self, in_put, length):
		embedded = self.embedding(in_put) # [batch, seq_len, embed_size]
		embedded = self.dropout(embedded)
		packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True,
														 enforce_sorted=False)
		poutput, (hidden_t, cell_t) = self.lstm(packed_embedded)

		output, output_length = nn.utils.rnn.pad_packed_sequence(poutput)

		if self.bidirectional:
			hidden = self.dropout(torch.cat([hidden_t[-1], hidden_t[-2]], dim=-1))
		else:
			hidden = self.dropout(hidden_t[-1])
		preds = self.fc(hidden)
		return preds


