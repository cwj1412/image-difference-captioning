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
    parser.add_argument('imagefolder', default='./image', help='Outfolder for images')
    parser.add_argument('outfile', help='outdata')
    args = parser.parse_args()

    datas = []
    index = {}

    count = 0
    imagelist = []
    with open(args.datafile, encoding='utf-8') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            if count == 0:
                print('Terms ', row)
            if count > 0:
                info1 = row[0].split('/')
                if info1[-1] not in imagelist:
                    download(row[1], args.imagefolder)
                    imagelist.append(info1[-1])
                info2 = row[4].split('/')
                if info2[-1] not in imagelist:
                    download(row[5], args.imagefolder)
                    imagelist.append(info2[-1])
            print('Images ', len(imagelist))
