import torch
import numpy as np
from transformers import BertTokenizer
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pretrained_model_dir = os.path.join(project_dir, "bert_base_uncased")
tokenizer = BertTokenizer.from_pretrained(pretrained_model_dir)

labels = {'business':0,
          'entertainment':1,
          'sport':2,
          'tech':3,
          'politics':4
          }

class Dataset(torch.utils.data.Dataset):
    def __init__(self,df):
        self.labels = [labels[label] for label in df['Category']]
        self.texts = [tokenizer(
            text,
            padding='max_length',
            max_length = 512,
            truncation=True,
            return_tensors="pt")
        for text in df['Text']]

    def classes(self):
        return  self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

class TestDataset(torch.utils.data.Dataset):
    def __init__(self,df):

        self.texts = [tokenizer(
            text,
            padding='max_length',
            max_length = 512,
            truncation=True,
            return_tensors="pt")
        for text in df['Text']]

    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        return batch_texts