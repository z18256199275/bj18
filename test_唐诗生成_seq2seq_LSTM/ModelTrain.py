# 训练模型
import torch, os
import torch.nn.functional as F
from torch import optim
from seq2seq import Seq2seq
from dataset import get_dataloader


def train(epoch):
    dataloader = get_dataloader(train=True)
    mySeq = Seq2seq()
    optimizer = optim.Adam(mySeq.parameters(), lr=0.001)

    if os.path.exists('./model/mySeq.pkl'):
        print('加载模型')
        mySeq.load_state_dict(torch.load('./model/mySeq.pkl'))
        optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))

    for idx,(input, target, input_len, target_len) in enumerate(dataloader):
        optimizer.zero_grad()
        # decoder_outputs [batch_size,max_len+1,vocab_size_output]
        decoder_outputs = mySeq(input, input_len)
        output = decoder_outputs.view(decoder_outputs.size(0)*decoder_outputs.size(1),-1)
        target = target.view(target.size(0)*target.size(1))
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()

        print('第{}个epoch的第{}个batch的loss={}'.format(epoch+1,idx+1,loss))

        if (idx+1) % 50 == 0:
            torch.save(mySeq.state_dict(),'./model/mySeq.pkl')
            torch.save(optimizer.state_dict(),'./model/optimizer.pkl')

if __name__ == '__main__':
    for i in range(3):
        train(i)





