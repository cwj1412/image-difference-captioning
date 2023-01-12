'''
    Evaluate results generateg by models
'''

import os
import json
import argparse
import eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ground truth for evalation')
    parser.add_argument('datafile', help='data file')
    # parser.add_argument('outfile', help='result file')
    args = parser.parse_args()

    candidates = {}
    references = {}

    with open(args.datafile, 'r') as fin:
        for line in fin:
            jterm = json.loads(line)
            candidates[jterm['ImgId']] = [' '.join(jterm['candidates'])]
            references[jterm['ImgId']] = [' '.join(s) for s in jterm['references']]
    print('Candidates ', len(candidates))
    print('References ', len(references))
    evaluator = eval.Evaluator(references, candidates)
    evaluator.evaluate()        
