# 使用LSTM完成编码，解码等工作
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
import config
from dataset import get_dataloader

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.ws_input),
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.ws_input.PAD)
        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            batch_first=True,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            bidirectional=config.bidirection)

    def forward(self, input, input_len):
        """
        :param input: [batch_size, seq_len]
        :param input_len: [batch_size]
        :return:
        """
        input_embedded = self.embedding(input)  # [batch_size, seq_len, embedding_dim]
        # 打包
        input_embedded_pack = pack_padded_sequence(input_embedded,lengths=input_len,batch_first=True)
        output,(h_n, c_n) = self.lstm(input_embedded_pack)
        # 解包
        out, out_len = pad_packed_sequence(output, batch_first=True, padding_value=config.ws_input.PAD)
        return out,h_n,c_n

if __name__ == '__main__':
    dataloader = get_dataloader()
    encoder = Encoder()
    for input, target, input_len, target_len in dataloader:
        _,h_n,c_n = encoder(input, input_len)
        print(h_n.size())  # [2*2,batch_size,hidden_size]
        print(c_n.size())
        break