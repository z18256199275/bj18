# 数据集整理，生成dataloader

from torch.utils.data import DataLoader, Dataset
import torch
import config

class CoupletDataset(Dataset):
    def __init__(self, train=True):
        input_path = config.train_data_path_in if train else config.test_data_path_in
        output_path = config.train_data_path_out if train else config.test_data_path_out

        data_input = open(input_path, encoding='utf8').readlines()
        data_output = open(output_path, encoding='utf8').readlines()
        assert len(data_input)==len(data_output), '输入输出长度不同'

        self.data_input = [sentence.strip().split() for sentence in data_input]
        self.data_output = [sentence.strip().split() for sentence in data_output]

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, index):
        input = self.data_input[index]
        input_len = len(input) if len(input) < config.max_len else config.max_len
        target = self.data_output[index]
        target_len = len(target) if len(target) < config.max_len+1 else config.max_len+1
        return input, target, input_len, target_len

def my_collate(batch):
    """
    :param batch: [(input,target,input_len,target_len),(),()]
    :return:
    """
    batch = list(sorted(batch, key=lambda x:x[3], reverse=True))
    print(batch)
    input, target, input_len, target_len = list(zip(*batch)) # [(input,input,input,...),(target,target,..),(input_len,...),(target_Len,...)]
    input = torch.LongTensor([config.ws_input.transform(sentence_list,max_len=config.max_len) for sentence_list in input])
    target = torch.LongTensor(
        [config.ws_output.transform(sentence_list, max_len=config.max_len, eos_add=True) for sentence_list in target])
    input_len = torch.LongTensor(input_len)
    target_len = torch.LongTensor(target_len)
    return input, target, input_len, target_len

def get_dataloader(train=True):
    MyDataset = CoupletDataset(train=train)
    batch_size = config.train_batch_size if train else config.test_batch_size
    dataloader = DataLoader(MyDataset, batch_size=2, shuffle=True, collate_fn=my_collate)
    return dataloader

if __name__ == '__main__':
    for input, target, input_len, target_len in get_dataloader():
        print(input.size())  #[batch_size, seq_len]
        print(target.size())  # [batch_size, seq_len+1]
        print(input_len.size())  #[batch_size]
        print(target_len.size())  # [batch_size]
        break
