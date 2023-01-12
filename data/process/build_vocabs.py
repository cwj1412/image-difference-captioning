import os
import json
import argparse
from nltk.tokenize import word_tokenize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('infile', type=str, help='input file')
    parser.add_argument('outfile', type=str, help='output file')
    parser.add_argument('dictfile', type=str, help='dict file')
    args = parser.parse_args()

    word_counts = {}
    datas = []
    flag = False
    theshold = 1 # min appearances
    if 'train' in args.infile:
        flag = True
    with open(args.infile, 'r', encoding='utf-8') as fin:
        for line in fin:
            jterm = json.loads(line.strip())
            for cap in jterm['description']:
                cap_tokens = word_tokenize(cap)
                if flag:
                    for w in cap_tokens:
                        word_counts[w] = word_counts.get(w, 0) + 1
                sample = {'img1': jterm['img1'],
                          'img2': jterm['img2'],
                          'description': cap_tokens}
                datas.append(sample)


    vocabs = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for w in word_counts:
        if word_counts[w] >= theshold:
            length = len(vocabs)
            vocabs[w] = length
    print('Vocab size ', len(vocabs))
    print('<BOS> ', vocabs['<BOS>'])

    
    print('Total Datas ', len(datas))
    with open(args.outfile, 'w', encoding='utf-8') as fout:
        for sample in datas:
            jterm = json.dumps(sample, ensure_ascii=False)
            fout.write(jterm+'\n')
    if flag:
        V = json.dumps(vocabs, ensure_ascii=False)
        with open(args.dictfile, 'w', encoding='utf-8') as fout:
            fout.write(V+'\n')





     