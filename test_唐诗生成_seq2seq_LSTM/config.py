import pickle

max_len = 20   # seq_len=20实际上就是指定每一个句子最长20个时间步
train_data_path_in = './couplet_data/train/in.txt'
train_data_path_out = './couplet_data/train/out.txt'

test_data_path_in = './couplet_data/test/in.txt'
test_data_path_out = './couplet_data/test/out.txt'

train_batch_size = 128
test_batch_size = 500

ws_input = pickle.load(open('./model/train_input.pkl','rb'))
ws_output = pickle.load(open('./model/train_target.pkl','rb'))

embedding_dim = 256
embedding_dim_output = 300
hidden_size = 100
num_layers = 2
bidirection = True
# encoder  双向，中输出h_n为[bach_szie,seq_len,2*hidden_size]
# 输入decoder中，因为此时为单向，因此num_layers得改变
decoder_num_layers = num_layers * 2 # 双向bidirectional 所以乘以2


