import torch
import torch.nn as nn
import torch.nn.functional as F

class BCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters=100, filter_sizes=[2, 3, 4], output_dim=4,
                 dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1,
                                              out_channels = n_filters,
                                              kernel_size = (fs, embedding_dim))
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)
    def forward(self, text):
        output = text.unsqueeze(1)
        conved = [F.relu(conv(output)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.fc(cat)
