"""
测试生成诗歌

训练数据生成模型，测试数据按照一个时间步一个时间步预测，
注意：测试时seq_len=1 并不等于训练模型时候的seq_len=48（此处），仍可以将输入放入模型，并不会报错
"""

"""
训练模型
"""
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from LstmModel import PoemLstmModel
from dataset import get_dataloader
import config

def generate(start_words):
    model = PoemLstmModel()
    if os.path.exists('./model/poemModel.pkl'):
        model.load_state_dict(torch.load('./model/poemModel.pkl'))
        print('加载一次模型')

    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = torch.Tensor([config.word2ix['<START>']]).view(1, 1).long()  # [1,1]  batch_size=seq_len=1

    # 最开始的隐状态初始为0矩阵 h_0,c_0所以为2  [num_layers*1, batch_Size, hidden_size]
    hidden = torch.zeros((2, config.num_layers * 1, 1, config.hidden_size), dtype=torch.float)
    model.eval()
    with torch.no_grad():
        for i in range(config.seq_len):  # 诗的长度
            output, hidden = model(input, hidden)  # [batch_size*seq_len, vocab_size]  [1,vocab_size]
            # print(output.shape)
            # 如果在给定的句首中，input为句首中的下一个字
            if i < start_words_len:
                w = results[i]
                input = input.data.new([config.word2ix[w]]).view(1, 1)
            # 否则将output作为下一个input进行
            else:
                top_index = output.data[0].topk(1)[1][0].item()  # 输出的预测的字
                w = config.ix2word[top_index]
                results.append(w)
                input = input.data.new([top_index]).view(1, 1)
            if w == '<EOP>':  # 输出了结束标志就退出
                del results[-1]
                break


        return results








