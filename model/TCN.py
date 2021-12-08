from torch import nn
from model.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, vocab_size):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Conv1d(num_channels[-1], vocab_size, kernel_size=1, bias=True)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        # self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1)