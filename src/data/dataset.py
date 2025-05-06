import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
import pathlib
import numpy as np
from os import path
import csv
import random

from src.utils.seeds import worker_init_fn, generator

class CalcDataset(torch.utils.data.Dataset):
    def __init__(self, data_num, word2id, train=True):
        super().__init__()

        self.data_num = data_num
        self.word2id = word2id
        self.train = train
        self.numbers = list("0123456789")
        self.operators = ['+']

        self.data, self.label = [], []
        for _ in range(data_num):
            x = int("".join([random.choice(self.numbers) for _ in range(random.randint(1, 3))])) # 0 ~ 999
            y = int("".join([random.choice(self.numbers) for _ in range(random.randint(1, 3))])) # 0 ~ 999
            op = random.choice(self.operators)
            left = ("{:*<7s}".format(str(x) + op + str(y))).replace("*", "<pad>")
            self.data.append(self.transform(left, seq_len=7))
            
            z = x + y
            right = ("{:*<6s}".format(str(z))).replace("*", "<pad>")
            right = self.transform(right, seq_len=5)
            right = [12] + right
            right[right.index(10)] = 12
            self.label.append(right)

        self.data = np.asarray(self.data)
        self.label = np.asarray(self.label)

    def __getitem__(self, index):
        X, y = self.data[index], self.label[index]
        return X, y
    
    def __len__(self):
        return self.data.shape[0]

    def transform(self, string, seq_len=7):
        tmp = []
        for index, char in enumerate(string):
            try:
                tmp.append(self.word2id[char])
            except:
                tmp += [self.word2id["<pad>"]] * (seq_len - index)
                break
        return tmp
    
def get_word2id():
    word2id = {str(i): i for i in range(10)}
    # word2id.update({"": 10, "+": 11, "": 12})
    word2id.update({"<pad>": 10, "+": 11, "<eos>": 12})
    return word2id

def get_id2word(word2id):
    id2word = {v: k for k, v in word2id.items()}
    return id2word
    
def create_dataset(train_data_num, test_data_num, batch_size, word2id):

    train_dataset = CalcDataset(train_data_num, word2id, train=True)
    test_dataset = CalcDataset(test_data_num, word2id, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    
    return train_loader, test_loader