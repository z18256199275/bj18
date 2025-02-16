import torch
import torch.nn as nn
import config
import torch.nn.functional as F

class PoemLstmModel(nn.Module):
    def __init__(self):
        super(PoemLstmModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.word2ix),
                                      embedding_dim=config.embedding_dim)

        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            batch_first=True,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers)

        self.fc1 = nn.Linear(config.hidden_size, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, len(config.word2ix))


    def forward(self, input, hidden=None):
        """
        :param input: [batch_size, seq_len]
        :return:
        """
        input = self.embedding(input)  # [batch_size, seq_len, embedding_dim]
        batch_size, seq_len = input.size(0), input.size(1)
        if hidden is None:
            h_0 = torch.zeros((config.num_layers*1, batch_size, config.hidden_size))
            c_0 = torch.zeros((config.num_layers*1, batch_size, config.hidden_size))

        else:
            h_0, c_0 = hidden

        output, hidden = self.lstm(input, (h_0,c_0))
        # output [batch_size, seq_len, hidden_size]
        # 经过几层全连接层
        output = torch.reshape(output, (batch_size*seq_len,-1))
        output = torch.tanh(self.fc1(output))  # [batch_size*seq_len, 2048]
        output = torch.tanh(self.fc2(output))  # [batch_size*seq_len, 4096]
        output = self.fc3(output) # [batch_size*seq_len, vocab_size]

        return F.log_softmax(output, dim=-1), hidden  # 用于计算损失


