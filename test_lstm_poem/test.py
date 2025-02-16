

from train import train
from eval import generate

if __name__ == '__main__':
    # for i in range(500):
    #     train(i)

    res = generate('雨')
    print(res)