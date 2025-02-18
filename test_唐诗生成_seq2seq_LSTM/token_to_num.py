# 实现中文单词到数值index的对应，同时实现transform等方法

class Token2num():

    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    SOS_TAG = 'SOS'
    EOS_TAG = 'EOS'

    UNK = 0
    PAD = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD,
            self.SOS_TAG:self.SOS,
            self.EOS_TAG:self.EOS
        }

        self.count = {}  # 放置 词:词频

    def fit(self,sentence):
        """
        传入句子，生成count词典，{词:词频} ，
        :param sentence:
        :return: {词:词频,....}
        """
        for word in sentence:
            self.count[word] = self.count.get(word,0) + 1

    # fit完成所有的语句之后，创建词典
    def build_vocab(self,min_count=5,max_count=None,max_feature=None):
        if min_count is not None:
            self.count = {k:v for k,v in self.count.items() if v>=min_count}
        if max_count is not None:
            self.count = {k:v for k,v in self.count.items() if v<=max_count}
        if max_feature is not None:  # 词典中只保存词频频率最高的前max_feature个词
            if len(self.count) > max_feature:
                self.count = dict(sorted(self.count.items(),key=lambda x:x[1],reverse=True)[:max_feature])

        # 根据上述筛选的词，构建词典
        for word in self.count:
            self.dict[word] = len(self.dict)

        # index:词
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    # 实现出入一个sentence,--> index
    def transform(self,sentence, max_len, eos_add=False):
        """
        实现词-->index转变
        :param sentence:[word1, word2,...]
        :param max_len: seq_len的最大长度
        :param eos_add: 是否添加EOS
        :return:
        """
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        length = len(sentence)
        if eos_add:
            sentence = sentence + [self.EOS_TAG]
        if length < max_len:
            sentence = sentence + (max_len-length)*[self.PAD_TAG]

        # 转化
        return [self.dict.get(word,self.UNK) for word in sentence]

    def __len__(self):
        return len(self.dict)

    # index --> sentence
    def inverse_transform(self, indices):
        result = [self.inverse_dict.get(index,self.UNK_TAG) for index in indices]
        # 由于result可能包含EOS，UNK等
        filter_result = []
        for word in result:
            if word == self.EOS_TAG:
                break
            filter_result.append(word)
        return ''.join(filter_result)


if __name__ == '__main__':
    ws = Token2num()
    ws.fit(['你好','我','是','一个','超人'])
    ws.fit(['平常','喜欢','飞','还是','非常','乐于助人','的'])
    ws.build_vocab(min_count=1, max_count=None, max_feature=100)
    print(ws.dict)
    print(ws.transform(['我','喜欢','飞'],10))