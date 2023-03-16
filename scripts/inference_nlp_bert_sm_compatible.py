import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, AutoConfig

import torch.nn as nn
import torch.nn.functional as F

import argparse
import logging
import os
import random
import sys

from datetime import datetime

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


TOKENIZER_MODEL = "bert-base-multilingual-uncased"
device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_model = "bert-base-multilingual-uncased"

# PyTorch models inherit from torch.nn.Module
class SentenceMultiClassClassifier(nn.Module):
    def __init__(self,number_class, pretrained_model):
        super(SentenceMultiClassClassifier, self).__init__()
        self.number_class = number_class
        #self.pretrained = AutoModel.from_pretrained(pretrained_model).to(device)
        self.pretrained = AutoModel.from_pretrained(pretrained_model,config=AutoConfig.from_pretrained(pretrained_model, output_attentions=True,output_hidden_states=True))

        #self.dropout = nn.Dropout(0.5) 
        #self.fc1 = nn.Linear(768, 1200)
        #self.fc2 = nn.Linear(1200, 1400)
        #self.fc3 = nn.Linear(1400, number_class)
        
        self.linear = nn.Linear(768, number_class)
        self.layeroutput = torch.nn.Sigmoid()



    def forward(self, input_ids, token_type_ids, attention_mask):            
        output_pretrained = self.pretrained(input_ids, token_type_ids, attention_mask)
        # Freeze the BERT parameters
        for param in self.pretrained.parameters():
            param.requires_grad = False
            
        #x = F.relu(self.fc1(output_pretrained.last_hidden_state[:,0,:].view(-1,768)))
        #x = self.dropout(x)
        #x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        
        x = output_pretrained.last_hidden_state[:,0,:].view(-1,768)
        x = self.linear(x)
        x = self.layeroutput(x)
        return x
    

def model_fn(model_dir):   
    NUM_CLASS = 11
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    model = SentenceMultiClassClassifier(NUM_CLASS,TOKENIZER_MODEL)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        #model = torch.jit.load(f)
        model.load_state_dict(torch.load(f))

    model.eval()
    return model.to(device), tokenizer

def predict_fn(text, model_and_tokenizer):
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer    
    
    # Tokenize sentences
    tokenized_text = tokenizer(text, max_length = 128, padding="max_length", return_tensors="pt")
    tokenized_text['input_ids'] = tokenized_text['input_ids'].to(device)
    tokenized_text['token_type_ids'] = tokenized_text['token_type_ids'].to(device)
    tokenized_text['attention_mask'] = tokenized_text['attention_mask'].to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**tokenized_text)

    # return dictonary, which will be json serializable
    return model_output.tolist()
