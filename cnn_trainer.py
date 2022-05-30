# author: Huihui Xu
# email: huihui.xu@pitt.edu
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from dataset import *
from cnn_pytorch import *
from loss_functions import *

import argparse
import functools
import sys
import os

from sklearn.metrics import classification_report


def accuracy(preds, y):
	max_preds = preds.argmax(dim=1, keepdim=True)
	correct = max_preds.squeeze(1).eq(y)
	return max_preds, correct.sum()/len(correct)

def data_iter(data, text_pipeline, label_pipeline):
	texts, labels, lengths = [], [], []
	for (text, label) in data:
		label = label_pipeline(label)
		text = text_pipeline(text)
		texts.append(torch.as_tensor(text))
		labels.append(torch.as_tensor(label))
		lengths.append(torch.as_tensor(len(text)))
	# convert to a dataframe
	df = pd.DataFrame({'sentence': texts, 'label': labels, 'length':lengths})
	records = df.to_dict('records')
	return records

def collate(batch, pad_index):
    batch_ids = [i['sentence'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = [i['length'] for i in batch]
    batch_length = torch.stack(batch_length)
    batch_label = [i['label'] for i in batch]
    batch_label = torch.stack(batch_label)
    batch = {'sentence': batch_ids,
             'length': batch_length,
             'label': batch_label}
    return batch



def trainer(model,
			train_dataloader,
			val_dataloader, 
			device,
			batch_size, 
			epochs, 
			learning_rate=2e-3, 
			weight_decay=0,
			loss='crossentropy'
			):
	
	if loss == 'crossentropy':
		criterion = nn.CrossEntropyLoss()
	elif loss == 'focal':
		criterion = FocalLoss()
	elif loss == 'f1':
		criterion = F1Loss()

	min_val_loss = float('inf')

	criterion = criterion.to(device)
	model = model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	# start train
	
	
	for epoch in range(epochs):
		train_epoch_loss = 0
		train_epoch_acc = 0
		
		print("start training...")
		model.train()
		for batch in train_dataloader:
			text = batch['sentence'].to(device)
			length = batch['length']
			label = batch['label'].to(device)
			optimizer.zero_grad()
		
			predictions = model(text)
			loss = criterion(predictions, label)
			# accuracy
			preds, acc = accuracy(predictions, label)
			loss.backward()
			optimizer.step()

			train_epoch_loss += loss.item()
			train_epoch_acc += acc.item()

		avg_train_loss = train_epoch_loss/len(train_dataloader)
		avg_train_acc = train_epoch_acc/len(train_dataloader)

		print("start validating...")
		val_epoch_loss = 0
		val_epoch_acc = 0

		model.eval()
		with torch.no_grad():
			for batch in val_dataloader:
				text = batch['sentence'].to(device)
				length = batch['length']
				label = batch['label'].to(device)

				predictions = model(text)
				loss = criterion(predictions, label)

				preds, acc = accuracy(predictions, label)

				val_epoch_loss += loss.item()
				val_epoch_acc += acc.item()
		avg_val_loss = val_epoch_loss/len(val_dataloader)
		avg_val_acc = val_epoch_acc/len(val_dataloader)	

		if avg_val_loss < min_val_loss:
			min_val_loss = avg_val_loss
			if not os.path.exists('saved_cnn_models'):
				os.mkdir('saved_cnn_models')

			torch.save(model, f'saved_cnn_models/tut-model.pt')

		print(f'Epoch: {epoch}')
		print(f'Training Loss: {avg_train_loss:.3f} | Training Accuracy: {avg_train_acc*100:.3f}%')
		print(f'Validation Loss: {avg_val_loss:.3f} | Valication Accuracy: {avg_val_acc*100:.3f}%')

def evaluate(model, device,  test_loader, loss = 'crossentropy'):
	if loss == 'crossentropy':
		criterion = nn.CrossEntropyLoss()
	elif loss == 'focal':
		criterion = FocalLoss()
	elif loss == 'f1':
		criterion = F1Loss()
	model = model.to(device)
	criterion = criterion.to(device)

	test_epoch_loss = 0
	test_epoch_acc = 0

	preds_list = []
	labels_list = []

	model.eval()
	with torch.no_grad():
		for batch in test_loader:
			text = batch['sentence'].to(device)
			length = batch['length']
			label = batch['label'].to(device)
			labels_list += batch['label']


			predictions = model(text)
			loss = criterion(predictions, label)
			preds, acc = accuracy(predictions,label)
			preds_list +=torch.flatten(preds).tolist()

			test_epoch_loss += loss.item()
			test_epoch_acc += acc.item()
	avg_test_loss = test_epoch_loss/len(test_loader)
	avg_test_acc = test_epoch_acc/len(test_loader)
	print(f'Test Loss: {avg_test_loss:.3f} | Test Accuracy: {avg_test_acc*100:.3f}%')	
	conf_matrix = classification_report(labels_list, preds_list)
	print('Confusion matrix:')
	print(conf_matrix)
	return preds		

	
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

if __name__ == "__main__":
	torch.manual_seed(0)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	my_parser = argparse.ArgumentParser()
	my_parser.add_argument("--data", type=str,
							default = 'summ')
	my_parser.add_argument("--epochs", type=int,
							default = 1)
	my_parser.add_argument("--batch_size", type=int,
							default = 64)
	my_parser.add_argument("--embed_size", type=int,
							default=300)
	my_parser.add_argument("--n_filters", type=int,
							default=100)
	my_parser.add_argument("--dropout_rate", type=float,
							default = 0.5)
	my_parser.add_argument("--lr", type=float,
							default = 2e-3)

	args = my_parser.parse_args()

	# load a tokenizer
	tokenizer = get_tokenizer('basic_english')

	# read and compile datasets
	custom_data = CustomDataset(file_type=args.data)
	train_dataset = custom_data.train_data
	valid_dataset = custom_data.valid_data
	test_dataset = custom_data.test_data

	# build vocabulary on training
	vocab = Vocabulary(tokenizer, train_dataset).vocabulary

	text_pipeline = lambda x:vocab(tokenizer(x))
	label_pipeline = lambda x: int(x)
	# formatted text
	train = data_iter(train_dataset, text_pipeline, label_pipeline)
	valid = data_iter(valid_dataset, text_pipeline, label_pipeline)
	test = data_iter(test_dataset, text_pipeline, label_pipeline)
	
	vocab_size = len(vocab)
	# load model
	model = CNN(vocab_size = vocab_size, embed_size= args.embed_size,
				output_dim=4, padding_idx = vocab['<pad>'],
				dropout=args.dropout_rate,
				n_filters = args.n_filters)
	# initialize weights
	model.apply(initialize_weights)
	# load text from fasttext
	vectors = torchtext.vocab.FastText()
	pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
	model.embedding.weight.data = pretrained_embedding

	collate = functools.partial(collate, pad_index = vocab['<pad>'])
	# construct dataloaders
	train_dataloader = torch.utils.data.DataLoader(train, 
												batch_size = args.batch_size,
												collate_fn = collate,
												shuffle=True)
	valid_dataloader = torch.utils.data.DataLoader(valid, 
												batch_size = args.batch_size,
												collate_fn=collate)
	test_dataloader = torch.utils.data.DataLoader(test, 
												batch_size = args.batch_size,
												collate_fn=collate)	

	# start training...
	trainer(model, train_dataloader, valid_dataloader,
			device = device,
			batch_size = args.batch_size, 
			epochs = args.epochs, 
			learning_rate=args.lr, 
			weight_decay=0)
	# test result
	evaluate(model, test_dataloader)




	


