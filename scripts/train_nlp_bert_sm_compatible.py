import pandas as pd 
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel, BertConfig

import torch.nn as nn
import torch.nn.functional as F

import argparse
import logging
import os
import random
import sys

from datetime import datetime

if 'SM_MODEL_DIR' in os.environ:
    pass
else:
    os.environ["SM_MODEL_DIR"] = "./"
    os.environ["SM_CHANNEL_TRAIN"] = "./"


class MyNLPDataset(Dataset):
    def __init__(self, file_name, model_name):
        # data loading
        df = pd.read_csv(file_name)
        self.data_en = df.loc[:,["Tweet"]].squeeze().values.tolist()
        self.label_index = torch.from_numpy(df.iloc[:,2:].astype(int).to_numpy())
        self.n_samples = df.shape[0]
        self.embedding_model = None
        self.tokenizer = None
        self.model_name = model_name
        self.tokenized_text = [ {} for i in range(df.shape[0])]
        #self.token_type_ids = None
        #self.attention_mask = None

    def __getitem__(self, index):
        if self.tokenized_text[index] == {}:
            if self.tokenizer == None:
                self.load_embedding_model()
            
            text = self.data_en[index]
            assert isinstance(text, str)
            text_tokenized = self.tokenize_function(text)
            self.tokenized_text[index] = text_tokenized
        #return self.tokenized_text[index]
        return self.tokenized_text[index], self.label_index[index]

    def __len__(self):
        return self.n_samples
    
    def load_embedding_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
    
    def tokenize_function(self,text):
        # generate token from dataset 
        tokenized_text = self.tokenizer(text, max_length = 128, padding="max_length", return_tensors="pt")
        tokenized_text['input_ids'] = torch.squeeze(tokenized_text['input_ids'])
        tokenized_text['token_type_ids'] = torch.squeeze(tokenized_text['token_type_ids'])
        tokenized_text['attention_mask'] = torch.squeeze(tokenized_text['attention_mask'])
        
        return tokenized_text
    
    
# PyTorch models inherit from torch.nn.Module
class SentenceMultiClassClassifier(nn.Module):
    def __init__(self,number_class, pretrained_model):
        super(SentenceMultiClassClassifier, self).__init__()
        self.number_class = number_class
        self.pretrained = BertModel.from_pretrained(pretrained_model)
        #self.pretrained = BertModel.from_pretrained(pretrained_model,config=AutoConfig.from_pretrained(pretrained_model, output_attentions=True,output_hidden_states=True))

        #self.dropout = nn.Dropout(0.5) 
        #self.fc1 = nn.Linear(768, 1200)
        #self.fc2 = nn.Linear(1200, 1400)
        #self.fc3 = nn.Linear(1400, number_class)
        
        self.linear = nn.Linear(768, number_class)
        self.layeroutput = torch.nn.Sigmoid()



    def forward(self, input_ids, token_type_ids, attention_mask):            
        output_pretrained = self.pretrained(input_ids, token_type_ids, attention_mask)
        # Freeze the BERT parameters
        #for param in self.pretrained.parameters():
        #    param.requires_grad = False
            
        #x = F.relu(self.fc1(output_pretrained.last_hidden_state[:,0,:].view(-1,768)))
        #x = self.dropout(x)
        #x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        #x = output_pretrained.last_hidden_state[:,0,:].view(-1,768)
        
        x = output_pretrained.pooler_output
        x = self.linear(x)
        x = self.layeroutput(x)
        return x

def train_one_epoch(epoch_index,training_loader, optimizer, model, loss_fn, device):
    running_loss = 0.
    last_loss = 0.
    batch_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        labels = labels.type(torch.FloatTensor)
        labels = labels.to(device)
        for key, value in inputs.items():
            inputs[key] = inputs[key].to(device)
            
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(**inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        batch_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            running_loss = 0.
           
    return batch_loss / len(training_loader)    




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--learning_rate", type=float, default=0.00001)

    # Data, model, and output directories
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    args, _ = parser.parse_known_args()

    print("Start training ...")
    NUM_CLASS = 11
    PRETRAINED_MODEL = args.model_id 
    BATCH_SIZE = args.train_batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    
    TRAIN_LOCATION = args.training_dir + "/sem_eval_2018_task_1_train.csv" 
    VALIDATION_LOCATION = args.training_dir + "/sem_eval_2018_task_1_validation.csv" 
    
    MODEL_SAVE_LOCATION = args.output_dir
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = MyNLPDataset(TRAIN_LOCATION, PRETRAINED_MODEL)
    validation_dataset = MyNLPDataset(VALIDATION_LOCATION, PRETRAINED_MODEL)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SentenceMultiClassClassifier(NUM_CLASS, PRETRAINED_MODEL)    
    model.to(device)    

    loss_fn = torch.nn.BCELoss()
    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0

    best_vloss = 1_000_000.
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, train_dataloader, optimizer, model, loss_fn, device)

        # We don't need gradients on to do reporting
        #model.train(False)
        model.eval()
        running_vloss = 0.0
        
        for i, vdata in enumerate(validation_dataloader):
            vinputs, vlabels = vdata
            vlabels = vlabels.type(torch.FloatTensor)
            vlabels = vlabels.to(device)
            for key, value in vinputs.items():
                vinputs[key] = vinputs[key].to(device)
            
            with torch.no_grad():
                voutputs = model(**vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = '{}/model.pth'.format(MODEL_SAVE_LOCATION)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    
