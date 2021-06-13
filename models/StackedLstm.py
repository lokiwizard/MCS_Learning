import torch
import torch.nn as nn
from utils.dataread import Generatedataset, Getdata
import copy

class StackedLstm(nn.Module):

    def __init__(self, input_size, layer_num, hidden_size, seq_length):
        super(StackedLstm, self).__init__()
        self.input_size = input_size
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.conv2d = nn.Conv2d(1, 1, (1, 1), 1)
        self.lstm = nn.LSTM(input_size=self.input_size, num_layers=self.layer_num,
                            hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size * self.seq_length*2, 2)

    def forward(self, x):
        x = x.reshape(-1, 1, self.seq_length, self.input_size)
        x = self.conv2d(x)
        x = x.reshape(-1, self.seq_length, self.input_size)
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, self.hidden_size * self.seq_length*2)
        return self.linear(x)


class TestModel(nn.Module):

    def __init__(self, input_size, layer_num, hidden_size, seq_length):
        super(TestModel, self).__init__()
        self.input_size = input_size
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=1,)
        self.lstm = nn.LSTM(input_size=input_size, num_layers=self.layer_num,
                            hidden_size=self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size * self.seq_length, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.reshape(-1, self.seq_length, self.input_size)
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, self.hidden_size * self.seq_length)
        return self.linear(x)


class TestModel_2(nn.Module):

    def __init__(self, input_size, layer_num, hidden_size, seq_length):
        super(TestModel_2, self).__init__()
        self.input_size = input_size
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=self.input_size, num_layers=self.layer_num,
                            hidden_size=self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size * self.seq_length, 2)

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, self.hidden_size * self.seq_length)
        return self.linear(x)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TestModel(input_size=2, layer_num=3, hidden_size=256, seq_length=10)
    net.to(device=device)
    seq_length = 10
    batch_size = 32
    data = Getdata("D:\dataset\gps_01-10\gps_20161101", nrows=8000)
    print(data.keys())
    seq_length = 10
    batch_size = 32
    dl_trains = list(Generatedataset(copy.deepcopy(data), seq_length, batch_size).values())[:1]
    for dl_train in dl_trains:
        for X, Y in dl_train:
            net(X)
            print(X.shape)
            break

