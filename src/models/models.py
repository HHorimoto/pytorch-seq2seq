import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, word2id, device):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2id["<pad>"])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, indices):
        embedding = self.word_embeddings(indices)
        if embedding.dim() == 2:
            embedding = torch.unsqueeze(embedding, 1)
        hx = torch.zeros(1, self.batch_size, self.hidden_dim, device=self.device)
        cx = torch.zeros(1, self.batch_size, self.hidden_dim, device=self.device)

        _, state = self.lstm(embedding, (hx, cx))
        return state


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, word2id, device):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2id["<pad>"])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, index, state):
        embedding = self.word_embeddings(index)
        if embedding.dim() == 2:
            embedding = torch.unsqueeze(embedding, 1)
        lstm_out, state = self.lstm(embedding, state)
        output = self.linear(lstm_out)
        return output, state
    