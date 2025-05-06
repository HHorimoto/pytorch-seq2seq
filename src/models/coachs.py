import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time

class Coach:
    def __init__(self, train_models, test_models, dataloaders, criterion, optimizers, word2id, id2word, device, num_epoch, num_debug):
        self.encoder_trian = train_models['encoder']
        self.decoder_train = train_models['decoder']
        self.encoder_test = test_models['encoder']
        self.decoder_test = test_models['decoder']
        self.train_loader = dataloaders['train']
        self.test_loader = dataloaders['test']
        self.criterion = criterion
        self.encoder_optimizer = optimizers['encoder']
        self.decoder_optimizer = optimizers['decoder']
        self.word2id = word2id
        self.id2word = id2word
        self.device = device
        self.num_epoch = num_epoch

        # store
        self.train_loss, self.test_loss = [], []
        self.test_acc = []
        self.num_debug = num_debug

    def _train_epoch(self):
        self.encoder_trian.train()
        self.decoder_train.train()
        dataloader = self.train_loader
        batch_loss = []
        
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            encoder_hidden = self.encoder_trian(X)
            source, target = y[:, :-1], y[:, 1:]
            decoder_hidden = encoder_hidden

            loss = 0
            for i in range(source.size(1)):
                decoder_output, decoder_hidden = self.decoder_train(source[:, i], decoder_hidden)
                decoder_output = torch.squeeze(decoder_output)
                loss += self.criterion(decoder_output, target[:, i])

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            batch_loss.append(loss.item())

        epoch_loss = np.nanmean(batch_loss)
        return epoch_loss

    def _test_epoch(self):
        self.encoder_test.load_state_dict(self.encoder_trian.state_dict())
        self.decoder_test.load_state_dict(self.decoder_train.state_dict())
        self.encoder_test.eval()
        self.decoder_test.eval()

        dataloader = self.test_loader
        batch_loss = []
        batch_accuracy = 0

        with torch.no_grad():
            for index, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                encoder_hidden = self.encoder_test(X)
                source, target = y[:, :-1], y[:, 1:]
                decoder_hidden = encoder_hidden

                loss = 0
                right = []
                for idx in range(source.size(1)):
                    decoder_output, decoder_hidden = self.decoder_test(source[:, idx], decoder_hidden)
                    decoder_output = torch.squeeze(decoder_output, 0)
                    # loss
                    loss += self.criterion(decoder_output, target[:, idx])
                    # acc
                    prob = F.softmax(torch.squeeze(decoder_output), dim=0)
                    max_idx = torch.argmax(prob.cpu().detach()).item()
                    token = self.id2word[max_idx]
                    if token == "<eos>":
                        break # continue
                    right.append(token)
                right = "".join(right)

                left = self._get_left_id2word(X)
                try:
                    right_int = int(right)
                    flag = eval(left) == right_int
                except:
                    flag = False

                if flag:
                    batch_accuracy += 1
                batch_loss.append(loss.item())

        epoch_loss = np.nanmean(batch_loss)
        epoch_acc = batch_accuracy / len(dataloader.dataset)
        return epoch_loss, epoch_acc
    
    def _get_left_id2word(self, data):
        x = list(data[0].cpu().detach().numpy())
        try:
            padded_idx_x = x.index(self.word2id["<pad>"])
        except ValueError:
            padded_idx_x = len(x)
        left = "".join(map(lambda char: str(self.id2word[char]), x[:padded_idx_x]))
        return left

    def train_test(self):
        start = time.time()
        for epoch in range(self.num_epoch):
            train_epoch_loss = self._train_epoch()
            test_epoch_loss, test_epoch_acc = self._test_epoch()

            print("epoch: ", epoch+1, "/", self.num_epoch)
            print("time: ", time.time()-start)
            print("[train] loss: ", train_epoch_loss)
            print("[test] loss: ", test_epoch_loss, ", acc: ", test_epoch_acc)

            self.train_loss.append(train_epoch_loss)
            self.test_loss.append(test_epoch_loss)
            self.test_acc.append(test_epoch_acc)