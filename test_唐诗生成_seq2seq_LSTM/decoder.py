# 使用LSTM实现解码

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from encoder import Encoder
from dataset import get_dataloader
import config

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.ws_output),
                                      embedding_dim=config.embedding_dim_output,
                                      padding_idx=config.ws_output.PAD)
        self.lstm = nn.LSTM(input_size=config.embedding_dim_output,
                            batch_first=True,
                            hidden_size=config.hidden_size,
                            num_layers=config.decoder_num_layers)
        self.fc = nn.Linear(config.hidden_size,len(config.ws_output))


    def forward(self,encoder_hidden,encoder_c):
        """
        :param encoder_hidden: 输入中最后一个时间步的隐藏状态
        :param encoder_c: 输入中最后一个时间步的细胞状态
        :return:
        """
        decoder_hidden = encoder_hidden
        decoder_c = encoder_c
        batch_size = decoder_hidden.size(1)
        # 初始化第一个时间步的输出 [batch_size, 1]
        decoder_input = torch.LongTensor(torch.ones([batch_size,1],dtype=torch.int64)*config.ws_input.SOS)

        decoder_outputs = torch.ones([batch_size,config.max_len+1,len(config.ws_output)])

        # 循环计算每一个时间步的输出(下一个时间步的输入)、隐藏状态、细胞状态
        for t in range(config.max_len+1):
            # decoder_output_t 是经过全连接 softmax处理后的[batch_size, vocab_size_output]
            decoder_output_t, decoder_hidden, decoder_c = self.forward_step(decoder_input, decoder_hidden, decoder_c)
            # 保存每一个时间步的输出
            decoder_outputs[:,t,:] = decoder_output_t
            value,index = torch.topk(decoder_output_t,k=1)
            decoder_input = index

        return decoder_outputs, decoder_hidden



    def forward_step(self,decoder_input,decoder_hidden,decoder_c):
        # 输入为[batch_szie, 1] ，需要先进行embedding
        embedded = self.embedding(decoder_input)  # [batch_size, 1, embedding_dim_output]
        # 经过lstm输出结果
        # decoder_output [batch_size,1,hidden_size]
        decoder_output, (decoder_hidden,decoder_c) = self.lstm(embedded, (decoder_hidden, decoder_c))
        # 每一个时间步都需要经过一个全连接，输出结果使用softmax求出最大的分类
        decoder_output = torch.squeeze(decoder_output)  # [batch_size,hidden_size]
        # print('decoder_output的形状为：',decoder_output.size())
        output = self.fc(decoder_output)  # [batch_size, vocab_size_output]
        # 经过softmax处理
        output = F.log_softmax(output,dim=-1)
        return output, decoder_hidden, decoder_c

    # ?模型评估
    def evaluate(self, encoder_hidden, encoder_c):
        decoder_hidden = encoder_hidden
        decoder_c = encoder_c
        batch_size = decoder_hidden.size(1)
        # 初始化第一个时间步的输出 [batch_size, 1]
        decoder_input = torch.LongTensor(torch.ones([batch_size, 1], dtype=torch.int64) * config.ws_input.SOS)

        indices = []
        for t in range(config.max_len+5):
            decoder_output_t, decoder_hidden, decoder_c = self.forward_step(decoder_input,decoder_hidden,decoder_c)
            value, index = torch.topk(decoder_output_t, 1)


            decoder_input = index
            indices.append(index.squeeze().detach().numpy())

        return indices

if __name__ == '__main__':
    encoder = Encoder()
    decoder = Decoder()

    for input,target,input_len,target_len in get_dataloader():
        _,encoder_hidden,encoder_c = encoder(input,input_len)
        decoder_outputs, decoder_hidden = decoder(encoder_hidden, encoder_c)
        indices = decoder.evaluate(encoder_hidden, encoder_c)
        print(np.array(indices).shape)
        print('decoder：',decoder_outputs.size())  # [batch_size,max_len+1,vocab_size]
        print(decoder_hidden.size())  #[4,batch_size,hidden_size]
        break