
import torch
from torch.utils.data import Dataset, DataLoader



class RoBERTData(Dataset):
	def __init__(self, df, tokenizer, max_len):
		self.tokenizer = tokenizer
		self.df = df
		self.max_len = max_len
		self.label_dict = {'Issue': 0, 'Reason':1, 'Conclusion':2, 'Non_IRC':3}
		self.df['IRC_type'] = self.df['IRC_type'].map(self.label_dict)
		self.sentences = self.df['sentence'].tolist()
		self.labels = self.df['IRC_type'].tolist()

	def __len__(self):
		return len(self.sentences)

	def __getitem__(self, index):
		text = str(self.sentences[index])

		encoded = self.tokenizer.encode_plus(
			text,
			None,
			add_special_tokens = True,
			max_length = self.max_len,
			pad_to_max_length = True,
			return_token_type_ids = True
			)
		ids = encoded['input_ids']
		mask = encoded['attention_mask']
		token_type_ids = encoded['token_type_ids']

		return {
			'ids': torch.tensor(ids, dtype = torch.long),
			'mask': torch.tensor(mask, dtype = torch.long),
			'token_type_ids': torch.tensor(token_type_ids, dtype = torch.long),
			'targets': torch.tensor(self.labels[index], dtype = torch.float)
		}
		

class BERTData(Dataset):
	def __init__(self, df, tokenizer, max_len):
		self.tokenizer = tokenizer
		self.df = df
		self.max_len = max_len
		self.label_dict = {'Issue': 0, 'Reason':1, 'Conclusion':2, 'Non_IRC':3}
		self.df['IRC_type'] = self.df['IRC_type'].map(self.label_dict)
		self.sentences = self.df['sentence'].tolist()
		self.labels = self.df['IRC_type'].tolist()

	def __len__(self):
		return len(self.sentences)

	def __getitem__(self, index):
		text = str(self.sentences[index])

		encoded = self.tokenizer.encode_plus(
			text,
			add_special_tokens = True,
			max_length = self.max_len, # truncate all the sentences
			truncation = True,
			padding = 'max_length',
			return_attention_mask = True,
			return_tensors='pt',
		)
		ids = encoded['input_ids'].squeeze(0)
		mask = encoded['attention_mask'].squeeze(0)
		token_type_ids = encoded['token_type_ids'].squeeze(0)

		return {
			'ids': torch.tensor(ids, dtype = torch.long),
			'mask': torch.tensor(mask, dtype = torch.long),
			'token_type_ids': torch.tensor(token_type_ids, dtype = torch.long),
			'targets': torch.tensor(self.labels[index], dtype = torch.float)
		}









