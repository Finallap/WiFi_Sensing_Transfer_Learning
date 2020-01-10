import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CSILSTM(nn.Module):
    def __init__(self, input_feature, hidden_size, num_class, num_layer=2):
        super().__init__()
        self.rnn = nn.LSTM(input_feature, hidden_size, num_layer)
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, inp):
        bs = inp.size()[1]
        if bs != self.bs:
            self.bs = bs
        e_out = self.e(inp)
        h0 = c0 = Variable(e_out.data.new(*(self.nl, self.bs, self.hidden_size)).zero_())
        rnn_o, _ = self.rnn(e_out, (h0, c0))
        rnn_o = rnn_o[-1]
        fc = F.dropout(self.fc2(rnn_o), p=0.8)
        return F.log_softmax(fc, dim=1)

        x, = self.rnn(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x