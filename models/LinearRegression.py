import torch
import torch.nn as nn

class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(5, 3)

    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    net = LinearRegression()


