# 测试模型的输出结果
import torch, os
from torch import nn
import numpy as np
from dataset import get_dataloader
from seq2seq import Seq2seq
from config import ws_output, ws_input

def eval():
    dataloader = get_dataloader(train=False)
    mySeq = Seq2seq()
    if os.path.exists('./model/mySeq.pkl'):
        mySeq.load_state_dict(torch.load('./model/mySeq.pkl'))
        print('加载模型')

    with torch.no_grad():
        for idx,(input,target,input_len,target_len) in enumerate(dataloader):
            indices = mySeq.evaluate(input, input_len)  # [seq_len+5,batch_size]
            indices = np.array(indices).transpose(1,0)

            input_b = input.numpy()
            input_before = [ws_input.inverse_transform(input_line) for input_line in input_b]
            target_b = target.numpy()
            target_before = [ws_output.inverse_transform(input_line) for input_line in target_b]
            pred_lists = [ws_output.inverse_transform(index_list) for index_list in indices]
            print(list(zip(input_before, target_before, pred_lists)))
            break

if __name__ == '__main__':
    eval()

