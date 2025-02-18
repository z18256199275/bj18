import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder
from dataset import get_dataloader

class Seq2seq(nn.Module):
    def __init__(self):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,input,input_len):
        encoder_output, encoder_hidden, encoder_c = self.encoder(input,input_len)
        decoder_outputs, decoder_hidden = self.decoder(encoder_hidden, encoder_c)
        return decoder_outputs

    def evaluate(self,input,input_len):
        encoder_output, encoder_hidden, encoder_c = self.encoder(input, input_len)
        indices = self.decoder.evaluate(encoder_hidden, encoder_c)
        return indices

if __name__ == '__main__':
    seq_my = Seq2seq()
    for input,target,input_len,target_len in get_dataloader():
        indices = seq_my.evaluate(input, input_len)
        print(torch.tensor(indices).size())  #[25,128]
        break