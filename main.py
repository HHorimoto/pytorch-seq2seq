import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image

from sklearn.metrics import accuracy_score

from src.utils.seeds import fix_seed
from src.data.dataset import create_dataset, get_word2id, get_id2word
from src.models.models import Encoder, Decoder
from src.models.coachs import Coach
from src.visualization.visualize import plot

def main():

    with open('config.yaml') as file:
        config_file = yaml.safe_load(file)
    print(config_file)
    
    NUM_EPOCH = config_file['config']['num_epoch']
    BATCH_SIZE = config_file['config']['batch_size']
    LR = config_file['config']['learning_rate']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    word2id = get_word2id()
    id2word = get_id2word(word2id)

    train_loader, test_loader = create_dataset(train_data_num=20000, test_data_num=200, batch_size=BATCH_SIZE, word2id=word2id)
    
    vocab_size = len(word2id)
    encoder_train = Encoder(vocab_size=vocab_size, embedding_dim=16, hidden_dim=128, 
                            batch_size=BATCH_SIZE, word2id=word2id, device=device).to(device)
    decoder_train = Decoder(vocab_size=vocab_size, embedding_dim=16, hidden_dim=128, 
                            batch_size=BATCH_SIZE, word2id=word2id, device=device).to(device)
    
    encoder_test = Encoder(vocab_size=vocab_size, embedding_dim=16, hidden_dim=128, 
                            batch_size=1, word2id=word2id, device=device).to(device)
    decoder_test = Decoder(vocab_size=vocab_size, embedding_dim=16, hidden_dim=128, 
                            batch_size=1, word2id=word2id, device=device).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=word2id["<pad>"])
    encoder_optimizer = optim.AdamW(encoder_train.parameters(), lr=LR)
    decoder_optimizer = optim.AdamW(decoder_train.parameters(), lr=LR)

    coach = Coach({'encoder': encoder_train, 'decoder': decoder_train}, {'encoder': encoder_test, 'decoder': decoder_test},
                  {'train': train_loader, 'test': test_loader}, criterion, {'encoder': encoder_optimizer, 'decoder': decoder_optimizer}, 
                  word2id, id2word, device, NUM_EPOCH, 3)
    coach.train_test()

    plot({'train': coach.train_loss, 'test': coach.test_loss}, 'loss')

if __name__ == "__main__":
    fix_seed()
    main()