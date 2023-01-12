'''
    Generate ground truth for evaluation
'''

import os
import json
import argparse
from eval import Evaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ground truth for evalation')
    parser.add_argument('datafile', help='data file')
    # parser.add_argument('outfile', help='result file')
    args = parser.parse_args()

    datas = {}
    candidates = {}
    references = {}
    
    with open(args.datafile, encoding='utf-8') as fin:
        for line in fin:
            jterm = json.loads(line.strip())
            img_id = jterm['img1']+'_'+jterm['img2']
            if img_id not in datas:
                datas[img_id] = []
            datas[img_id].append(jterm['description'])
    for img_id in datas:
        for i in range(0, len(datas[img_id])):
            candidates[img_id+'_'+str(i)] = [' '.join(datas[img_id][i])]
            references[img_id+'_'+str(i)] = [' '.join(s) for s in datas[img_id][0:i]+datas[img_id][i+1:]]
    #print(candidates["23628656_2866428_0"])
    #print(references["23628656_2866428_1"])
    print('Candidates ', len(candidates))
    print('References ', len(references))
    evaluator = Evaluator(references, candidates)
    evaluator.evaluate()
