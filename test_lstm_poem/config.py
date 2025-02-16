import pickle

seq_len = 48
batch_size = 16
embedding_dim = 128
hidden_size = 256
num_layers = 2


word2ix = pickle.load(open('./corpus/word2ix.pkl','rb'))
ix2word = pickle.load(open('./corpus/ix2word.pkl','rb'))