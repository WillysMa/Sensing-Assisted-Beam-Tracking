import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GruModelSimple(nn.Module):
    def __init__(self, num_classes, num_layers=1, hidden_size=64, embed_size=64):
        super(GruModelSimple, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,embed_size),
        )
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.8,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.name = "GruModelSimple"
        # self.dropout1 = nn.Dropout(0.5)


    def forward(self, x):
        y = self.embed(x)
        # y = self.dropout1(y)
        y, _ = self.gru(y)
        y = self.fc(y)
        return y


class GruModelSimple_v1(nn.Module):
    def __init__(self, num_classes, num_layers=1, hidden_size=64, embed_size=64):
        super(GruModelSimple_v1, self).__init__()
        self.embed = torch.nn.Linear(4, embed_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.8,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.name = "GruModelSimple_v1"
        # self.dropout1 = nn.Dropout(0.5)


    def forward(self, x):
        y = self.embed(x)
        # y = self.dropout1(y)
        y, _ = self.gru(y)
        y = self.fc(y)
        return y