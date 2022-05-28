
import torch
import torch.nn as nn

# an implementation of Kim 2014 CNN model
# heavily referce from https://github.com/bentrevett/pytorch-sentiment-analysis
class CNN(nn.Module):
	def __init__(self, vocab_size, embed_size,
				output_dim, padding_idx, n_filters = 100, filter_sizes=[3,4,5], dropout = 0.5):
		super(CNN, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
		self.convs = nn.ModuleList([nn.Conv1d(embed_size, n_filters, filter_size) for filter_size in filter_sizes])
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)

	def forward(self, in_put):
		embedded = self.embedding(in_put) #[batch, seq_len, embed_size]
		embedded = self.dropout(embedded).permute(0,2,1)
		convs = [torch.relu(conv(embedded))for conv in self.convs]
		pooled = [conv.max(dim=-1).values for conv in convs]

		cat = self.dropout(torch.cat(pooled, dim=-1))
		preds = self.fc(cat)
		return preds





