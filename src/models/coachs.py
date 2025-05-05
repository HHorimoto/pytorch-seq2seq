import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time

class CoachTrain:
    def __init__(self, encoder, decoder, dataloader, criterion, 
                 encoder_optimizer, decoder_optimizer, device, num_epoch):
        self.encoder = encoder
        self.decoder = decoder
        self.dataloader = dataloader
        self.criterion = criterion
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer =decoder_optimizer
        self.device = device
        self.num_epoch = num_epoch

        # store
        self.train_loss = []

    def _train_epoch(self):
        self.encoder.train()
        self.decoder.train()
        batch_loss = []
        
        for X, y in self.dataloader:
            X, y = X.to(self.device), y.to(self.device)
            encoder_hidden = self.encoder(X)
            source, target = y[:, :-1], y[:, 1:]
            decoder_hidden = encoder_hidden

            loss = 0
            for i in range(source.size(1)):
                decoder_output, decoder_hidden = self.decoder(source[:, i], decoder_hidden)
                decoder_output = torch.squeeze(decoder_output)
                loss += self.criterion(decoder_output, target[:, i])

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            batch_loss.append(loss.item())

        epoch_loss = np.mean(batch_loss)
        return epoch_loss
    
    def train(self):
        start = time.time()
        for epoch in range(self.num_epoch):
            train_epoch_loss = self._train_epoch()

            print("epoch: ", epoch+1, "/", self.num_epoch)
            print("time: ", time.time()-start)
            print("[train] loss: ", train_epoch_loss)

            self.train_loss.append(train_epoch_loss)

class CoachTest:
    def __init__(self, encoder, decoder, dataloader, word2id, id2word, device):
        self.encoder = encoder
        self.decoder = decoder
        self.dataloader = dataloader
        self.word2id = word2id
        self.id2word = id2word
        self.device = device

        # store
        self.test_acc = []

    def test(self):
        self.encoder.eval()
        self.decoder.eval()
        accuracy = 0

        with torch.no_grad():
            for X, y in self.dataloader:
                X, y = X.to(self.device), y.to(self.device)
                state = self.encoder(X)

                right = []
                token = "<eos>"
                for _ in range(7):
                    index = self.word2id[token]
                    input_tensor = torch.tensor([index], device=self.device)
                    output, state = self.decoder(input_tensor, state)
                    prob = F.softmax(torch.squeeze(output), dim=0) 
                    index = torch.argmax(prob.cpu().detach()).item() # max prob id idex
                    token = self.id2word[index]
                    if token == "<eos>":
                        break
                    right.append(token)
                right = "".join(right)
                
                x = list(X[0].cpu().detach().numpy())
                try:
                    padded_idx_x = x.index(self.word2id["<pad>"])
                except ValueError:
                    padded_idx_x = len(x)
                left = "".join(map(lambda char: str(self.id2word[char]), x[:padded_idx_x]))
                
                # acc flag
                try:
                    right_int = int(right) 
                    flag = eval(left) == right_int
                except:
                    flag = False

                if flag:
                    accuracy += 1

        self.test_acc = accuracy / len(self.dataloader.dataset)