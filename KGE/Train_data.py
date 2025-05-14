import pandas as pd
import numpy as np
import sys

def main():
    data_file = '../data/knowgraph_all.tsv'
    df = pd.read_csv(data_file, sep="\t")
    triples = df.values.tolist()
    print(triples[0][0])
    num_triples = len(triples)
    print(num_triples)

    seed = np.arange(num_triples)
    np.random.shuffle(seed)

    train_cnt = int(num_triples * 0.9)
    valid_cnt = int(num_triples * 0.05)
    train_set = seed[:train_cnt]
    train_set = train_set.tolist()
    valid_set = seed[train_cnt:train_cnt + valid_cnt].tolist()
    test_set = seed[train_cnt + valid_cnt:].tolist()

    with open("../data/train/data_train.tsv", 'w+') as f:
        for idx in train_set:
            f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))

    with open("../data/train/data_valid.tsv", 'w+') as f:
        for idx in valid_set:
            f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))

    with open("../data/train/data_test.tsv", 'w+') as f:
        for idx in test_set:
            f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))


if __name__ == '__main__':
        main()
