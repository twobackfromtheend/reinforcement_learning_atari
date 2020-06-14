import torch
from torch import nn


class LSTMCartPoleModel(nn.Module):
    def __init__(self, inputs, outputs, device):
        super().__init__()
        self.device = device
        self.lstm_layers = 1
        self.lstm_hidden = 64
        self.lstm_1 = nn.LSTM(inputs, self.lstm_hidden, num_layers=self.lstm_layers)

        self.lin_1 = nn.Linear(self.lstm_hidden, 64)
        self.relu_1 = nn.ReLU()
        self.lin_2 = nn.Linear(self.lstm_hidden, 64)
        self.relu_2 = nn.ReLU()
        self.lin_out = nn.Linear(64, outputs)

    def initial_hidden(self, batch_size):
        return (torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=self.device),
                torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=self.device))

    def forward(self, x, hidden):
        # print(x.size())
        x, hidden = self.lstm_1(x, hidden)
        x = self.lin_1(x)
        x = self.relu_1(x)
        x = self.lin_2(x)
        x = self.relu_2(x)
        x = self.lin_out(x)

        return x, hidden


class LSTMAtariModel(nn.Module):
    def __init__(self, h, w, c, outputs, device):
        super().__init__()
        self.device = device

        kernel_size = 8
        stride = 2

        self.conv1 = nn.Conv2d(c, 16, kernel_size=kernel_size, stride=stride)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=stride)
        # self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=kernel_size, stride=stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        _h = h
        _w = w
        for i in range(2):
            _h, _w = conv2d_size_out(_h, kernel_size, stride), conv2d_size_out(_w, kernel_size, stride)
            print(_h, _w)
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # self.linear_input_size = convh * convw * 32
        self.linear_input_size = _h * _w * 32
        # print(self.linear_input_size)
        self.lin_1 = nn.Linear(self.linear_input_size, 256)
        self.relu_1 = nn.ReLU()

        self.lstm_layers = 1
        self.lstm_hidden = 128
        self.lstm_1 = nn.LSTM(256, self.lstm_hidden, num_layers=self.lstm_layers)

        self.lin_2 = nn.Linear(self.lstm_hidden, 64)
        self.relu_2 = nn.ReLU()
        self.lin_3 = nn.Linear(64, outputs)

    def initial_hidden(self, batch_size):
        return (torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=self.device),
                torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=self.device))

    def forward(self, x, hidden):
        # print(x.shape)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        # x = nn.functional.relu(self.bn1(self.conv1(x)))
        # x = nn.functional.relu(self.bn2(self.conv2(x)))
        # x = nn.functional.relu(self.bn3(self.conv3(x)))

        # print(x.shape, "asda")
        x = x.reshape((-1, 1, self.linear_input_size))
        # print(x.shape, "kfje")

        x = self.relu_1(self.lin_1(x))
        # print(x.shape, "grersw")
        x, hidden = self.lstm_1(x, hidden)
        x = self.lin_2(x)
        x = self.relu_2(x)
        x = self.lin_3(x)

        return x, hidden
