import os
import json
import argparse
from nltk.tokenize import word_tokenize

def tokenize_captions(captions):
    tokenize_captions = []
    for c in captions:
        tokenize_captions.append(word_tokenize(c))
    return tokenize_captions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tokenize captions for test')
    parser.add_argument('infile', type=str, help='input file')
    parser.add_argument('outfile', type=str, help='output file')
    args = parser.parse_args()

    with open(args.outfile, 'w', encoding='utf-8') as fout:
        with open(args.infile, 'r', encoding='utf-8') as fin:
            for line in fin:
                data = json.loads(line.strip())
                data['description'] = tokenize_captions(data['description'])
                jterm = json.dumps(data, ensure_ascii=False)
                fout.write(jterm+'\n') 