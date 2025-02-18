# 准备语料，注意输入和输出是不同的语料，可以分开生成不同的词典

import config
from token_to_num import Token2num
import pickle

if __name__ == '__main__':
    # 读取训练数据输入文件，生成输入语料词典  ，按照字进行切分
    train_input = open(config.train_data_path_in, encoding='utf8').readlines()
    train_target = open(config.train_data_path_out, encoding='utf8').readlines()
    # print(type(train_input),len(train_input))
    # print(len(train_target))

    train_input = [sentence.strip().split() for sentence in train_input]
    ws = Token2num()
    for sentence_list in train_input:
        ws.fit(sentence_list)
    ws.build_vocab()
    pickle.dump(ws,open('./model/train_input.pkl','wb'))

    train_target = [sentence.strip().split() for sentence in train_target]
    ws = Token2num()
    for sentence_list in train_target:
        ws.fit(sentence_list)
    ws.build_vocab()
    pickle.dump(ws,open('./model/train_target.pkl','wb'))

    print('保存完成')
