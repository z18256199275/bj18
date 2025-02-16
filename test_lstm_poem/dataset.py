"""
构造诗歌数据集合
# </s>:8292  <EOP>:8290  <START>:8291
每一条数据都是  [8292,..8292,8291,诗词Index,...,诗词index,8290]
原始诗歌长度为125，包含大量空值

"""
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch
import pickle as pkl
import config

class PoemDataSet(Dataset):
    def __init__(self,poem_path,seq_len):
        self.seq_len = seq_len
        self.poem_path = poem_path
        self.poem_data, self.ix2word, self.word2ix = self.get_raw_data()
        self.no_space_data = self.filter_space()

    def __getitem__(self,index):
        # 不在区分一首诗是一首诗，而是区分哪些字一起出现
        txt = self.no_space_data[index*self.seq_len : (index+1)*self.seq_len]
        label = self.no_space_data[index*self.seq_len + 1 : (index+1)*self.seq_len + 1]
        txt = torch.LongTensor(torch.tensor(txt, dtype=torch.int64))
        label = torch.LongTensor(torch.tensor(label, dtype=torch.int64))
        return txt, label

    def __len__(self):
        return int(len(self.no_space_data) / self.seq_len)

    def filter_space(self):  #
        t_data = torch.from_numpy(self.poem_data).view(-1)  # [7197500]
        flat_data = t_data.numpy()
        no_space_data = []
        for i in flat_data:
            if (i != 8292):
                no_space_data.append(i)
        return no_space_data

    def get_raw_data(self):
        datas = np.load(self.poem_path,allow_pickle=True)
        data = datas['data']
        ix2word = datas['ix2word'].item()
        word2ix = datas['word2ix'].item()

        return data, ix2word, word2ix

dataset = PoemDataSet('corpus/tang.npz', config.seq_len)
# word2ix = dataset.word2ix
# ix2word = dataset.ix2word
# pkl.dump(word2ix, open('./corpus/word2ix.pkl', 'wb'))
# pkl.dump(ix2word, open('./corpus/ix2word.pkl', 'wb'))

def get_dataloader(batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == '__main__':
    dataloader = get_dataloader(2)
    for input, target in dataloader:
        print(input.shape)  # [batch_size, seq_len]
        print(target.shape)
        break
