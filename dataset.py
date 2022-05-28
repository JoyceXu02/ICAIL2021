
import pandas as pd

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class CustomDataset:
	def __init__(self, file_type='summ'):
		# define data field
		# self.name = data.Field()
		# self.sentence = data.Field(tokenize = 'spacy', batch_first=True)
		# self.label = data.LabelField(sequential = False)
		self.file_type = file_type
		self.label_dict = {'Issue': 0, 'Reason':1, 'Conclusion':2, 'Non_IRC':3}
		# read and map data
		self.train_data, self.valid_data, self.test_data = self.read_data(self.file_type)

		# skip the position column
		# self.fields = [('name', self.name), (None, None), ('sentence', self.sentence), ('label', self.label)]


	def read_data(self, file_type):
		if file_type == 'summ':
			train = 'data/summ_train.csv'
			valid = 'data/summ_validation.csv'
			test = 'data/summ_test.csv'
		else:
			train = 'data/full_articles_train.csv'
			valid = 'data/full_articles_validation.csv'
			test = 'data/full_articles_test.csv'

		train_data = pd.read_csv(train, delimiter=',')
		test_data = pd.read_csv(test, delimiter=',')
		valid_data = pd.read_csv(valid, delimiter=',')

		# apply label_list
		train_data['IRC_type'] = train_data['IRC_type'].map(self.label_dict)
		test_data['IRC_type'] = test_data['IRC_type'].map(self.label_dict)
		valid_data['IRC_type'] = valid_data['IRC_type'].map(self.label_dict)

		train_tuples = train_data[['sentence', 'IRC_type']].to_records(index=False)
		test_tuples = test_data[['sentence', 'IRC_type']].to_records(index=False)
		valid_tuples = valid_data[['sentence', 'IRC_type']].to_records(index=False)

		return train_tuples, valid_tuples, test_tuples


class Vocabulary:
	def __init__(self, tokenizer, train_data):
		self.tokenizer = tokenizer
		self.train_data = train_data
		self.vocabulary = self.build_vocab()

	def yield_tokens(self):
		for text, _ in self.train_data:
			yield self.tokenizer(text)

	def build_vocab(self):
		vocab = build_vocab_from_iterator(self.yield_tokens(), specials=['<unk>', '<pad>'])
		vocab.set_default_index(vocab['<unk>'])
		return vocab


