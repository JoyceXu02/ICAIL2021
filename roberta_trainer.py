# author: Huihui Xu
# email: huihui.xu@pitt.edu
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import argparse
import os
import pandas as pd
from bert_dataset import *
from loss_functions import *

import transformers
from transformers import RobertaModel, RobertaTokenizer

from roberta_pytorch import *

from sklearn.metrics import classification_report, f1_score


def accuracy(preds, y):
	max_preds, max_idx = torch.max(preds.data, dim = 1)
	num_correct = (max_idx == y).sum()
	return max_idx, num_correct/len(preds)



def trainer(model, device, train_dataloader, val_dataloader, epochs, batch_size, lr, loss = 'crossentropy'):
	if loss == 'crossentropy':
		criterion = nn.CrossEntropyLoss()
	elif loss == 'focal':
		criterion = FocalLoss()
	elif loss == 'f1':
		criterion = F1Loss()

	criterion = criterion.to(device)
	model = model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr)

	min_val_loss = float('inf')

	for epoch in range(epochs):
		train_epoch_loss = 0
		train_epoch_acc = 0
		print("start training...")
		model.train()

		for i, data in enumerate(train_dataloader):
			ids = data['ids'].to(device, dtype = torch.long)
			mask = data['mask'].to(device, dtype = torch.long)
			token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
			targets = data['targets'].to(device, dtype = torch.long)

			outputs = model(ids, mask, token_type_ids)
			loss = criterion(outputs, targets)			
			max_idx, acc = accuracy(outputs, targets)
			train_epoch_loss += loss.item()
			train_epoch_acc += acc.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		avg_train_loss = train_epoch_loss/len(train_dataloader)
		avg_train_acc = train_epoch_acc/len(train_dataloader)
		print("start validating...")
		model.eval()
		val_epoch_loss = 0
		val_epoch_acc = 0
		for i, data in enumerate(val_dataloader):
			ids = data['ids'].to(device, dtype = torch.long)
			mask = data['mask'].to(device, dtype = torch.long)
			token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
			targets = data['targets'].to(device, dtype = torch.long)

			outputs = model(ids, mask, token_type_ids)
			loss = criterion(outputs, targets)			
			max_idx, acc = accuracy(outputs, targets)
			val_epoch_loss += loss.item()
			val_epoch_acc += acc.item()

		avg_val_loss = val_epoch_loss/len(val_dataloader)
		avg_val_acc = val_epoch_acc/len(val_dataloader)	

		if avg_val_loss < min_val_loss:
			min_val_loss = avg_val_loss
			if not os.path.exists('saved_roberta_models'):
				os.mkdir('saved_roberta_models')
			torch.save(model, f'saved_roberta_models/tut-model.pt')
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
		for i, data in enumerate(test_loader):
			ids = data['ids'].to(device, dtype = torch.long)
			mask = data['mask'].to(device, dtype = torch.long)
			token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
			targets = data['targets'].to(device, dtype = torch.long)
			labels_list += targets.tolist()

			outputs = model(ids, mask, token_type_ids)
			loss = criterion(outputs, targets)			
			max_idx, acc = accuracy(outputs, targets)
			test_epoch_loss += loss.item()
			test_epoch_acc += acc.item()
			preds_list +=torch.flatten(max_idx).tolist()


	avg_test_loss = test_epoch_loss/len(test_loader)
	avg_test_acc = test_epoch_acc/len(test_loader)
	print(f'Test Loss: {avg_test_loss:.3f} | Test Accuracy: {avg_test_acc*100:.3f}%')

	conf_matrix = classification_report(labels_list, preds_list)
	print('Confusion matrix:')
	print(conf_matrix)
	return preds_list		


if __name__ == "__main__":
	
	my_parser = argparse.ArgumentParser()
	my_parser.add_argument("--pretrained", type=str,
							default = 'roberta-base')
	my_parser.add_argument("--data", type=str,
							default = 'summ')
	my_parser.add_argument("--max_len", type = int,
							default = 256)
	my_parser.add_argument("--epochs", type=int,
							default = 3)
	my_parser.add_argument("--batch_size", type=int,
							default = 16)
	my_parser.add_argument("--dropout_rate", type=float,
							default = 0.5)
	my_parser.add_argument("--lr", type=float,
							default = 2e-5),
	my_parser.add_argument("--loss", type=str,
							default = 'crossentropy')

	args = my_parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# load Roberta tokenizer 
	tokenizer = RobertaTokenizer.from_pretrained(args.pretrained, truncation=True, do_lower_case=True)

	train_path, test_path, valid_path = None, None, None
	if args.data == 'summ':
		train_path = f'data/{args.data}_train.csv'
		test_path = f'data/{args.data}_test.csv'
		valid_path = f'data/{args.data}_validation.csv'
	else:
		train_path = f'data/{args.data}_articles_train.csv'
		test_path = f'data/{args.data}_articles_test.csv'
		valid_path = f'data/{args.data}_articles_validation.csv'
	# read training, test, and validation
	train_df = pd.read_csv(train_path, delimiter = ',')
	test_df = pd.read_csv(test_path, delimiter = ',')
	valid_df = pd.read_csv(valid_path, delimiter = ',')

	# construct RoBERT datasets
	training_set = RoBERTData(train_df, tokenizer, max_len = args.max_len)
	test_set = RoBERTData(test_df, tokenizer, max_len = args.max_len)
	valid_set = RoBERTData(valid_df, tokenizer, max_len = args.max_len)

	train_params = {'batch_size': args.batch_size,
					'shuffle': True,
					'num_workers': 0}

	test_params = {'batch_size': args.batch_size,
					'shuffle': False,
					'num_workers': 0}

	valid_params = {'batch_size': args.batch_size,
					'shuffle': False,
					'num_workers': 0}
	# construct dataloaders
	train_loader = DataLoader(training_set, **train_params)
	test_loader = DataLoader(test_set, **test_params)
	valid_loader = DataLoader(valid_set, **valid_params)

	# load model
	print("loading model...")
	model = Roberta(pretrained = args.pretrained)

	# model, device, criterion, train_dataloader, valid_dataloader, epochs, batch_size, lr, loss
	print("start training...")
	trainer(model= model, device=device, 
			train_dataloader= train_loader, val_dataloader=valid_loader,  
			epochs = args.epochs, batch_size = args.batch_size, lr = args.lr, loss = args.loss)

	print('load the best model...')
	best_model = torch.load('saved_roberta_models/tut-model.pt',map_location={'cuda:1':'cuda:0'})
	preds = evaluate(best_model, device, test_loader, loss=args.loss)
















