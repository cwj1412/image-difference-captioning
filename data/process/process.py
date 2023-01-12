'''
    Download images from Internet
'''

import os
import csv
import json
import argparse

def download(link, outfolder):
    url = link.split('?')
    info = link.split('/')
    outfile = os.path.join(outfolder, info[4]+'.jpg')
    if not os.path.exists(outfile):
        cmd = 'wget '+link+' -O '+outfile
        print(cmd)
        os.system(cmd)
    return info[4]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download bird images')
    parser.add_argument('datafile', help='Bird to words data')
    parser.add_argument('set', help='train/test/val')
    parser.add_argument('outfolder', help='outdata')
    args = parser.parse_args()

    outfile = os.path.join(args.outfolder, args.set+'.json')

    datas = {}

    count = 0
    with open(args.datafile, encoding='utf-8') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            if count == 0:
                print('Terms ', row)
            if count > 0:
                info1 = row[0].split('/')
                info2 = row[4].split('/')
                imgID = info1[-1]+'_'+info2[-1]
                if row[8] !=  args.set:
                    continue
                if imgID in datas:
                    datas[imgID]['description'].append(row[10])
                else:
                    sample = {'img1ImgURL': row[0],
                              'img2ImgURL': row[4],
                              'img1': info1[-1],
                              'img2': info2[-1],
                              'split': row[8],
                              'annN': row[9],
                              'description': [row[10]]}
                    datas[imgID] = sample
            count += 1
    print(count)
    total = 0
    with open(outfile, 'w', encoding='utf-8') as fout:
        for key in datas:
            total += len(datas[key]['description'])
            jterm = json.dumps(datas[key], ensure_ascii=False)
            fout.write(jterm+'\n')
    print(total)