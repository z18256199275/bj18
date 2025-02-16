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

def train(epoch):
    model = PoemLstmModel()
    optimizer = Adam(model.parameters(), lr=0.0001)
    if os.path.exists('./model/poemModel.pkl'):
        model.load_state_dict(torch.load('./model/poemModel.pkl'))
        optimizer.load_state_dict(torch.load('./model/poemOptim.pkl'))
        print('加载一次模型')

    dataloader = get_dataloader(config.batch_size)
    loop = tqdm(enumerate(dataloader), total=len(dataloader))

    for idx, (input, target) in loop:
        optimizer.zero_grad()
        # output [batch_size*seq_len, vocab_size]
        output, hidden = model(input)  # 训练的时候不用加hidden_state，训练中会自动传入hidden
        target_view = target.view(-1)  # [batch_size*seq_len)
        loss = F.nll_loss(output, target_view)
        loss.backward()
        optimizer.step()

        # 计算中准确率
        batch_size = input.size(0)
        seq_len = input.size(1)
        acc_output = output.reshape((batch_size, seq_len, -1))  # [batch_size, seq_len, vocab_size]
        pred = torch.topk(acc_output, dim=-1, k=1)[-1]
        pred = pred.squeeze()  # [batch_size, seq_len]
        acc = torch.eq(pred, target).float().mean()


        loop.set_description('[EPOCH]={},idx={},loss={},acc={}'.format(epoch,idx,loss,acc))
        if (idx+1) % 20 == 0:
            torch.save(model.state_dict(),'./model/poemModel.pkl')
            torch.save(optimizer.state_dict(),'./model/poemOptim.pkl')








