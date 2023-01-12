import os
import json
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize


def load_images(data_path):
    images = {}
    img_names = os.listdir(data_path)
    for i in tqdm(range(len(img_names))):
        if 'test2014' in img_names[i]:
            continue
        image_path = os.path.join(data_path, img_names[i])
        data = np.load(image_path)
        images[img_names[i].replace('.npy', '')] = data
    print('Total images ', len(images))
    return images

def load_vocabs(data_path):
    dict_path = os.path.join(data_path, 'dict.json')
    try:
        word2id = json.load(open(dict_path, 'r', encoding='utf-8'))
    except:
        word2id = build_vocabs(data_path)
    id2word = {word2id[key]:key for key in word2id}
    print('Vocabulary Size', len(word2id))
    return word2id, id2word

def build_vocabs(data_path):
    train_path = os.path.join(data_path, 'train.json')
    datas = json.load(open(train_path, 'r', encoding='utf-8'))
    word_counts = {}
    theshold = 1 # min appearances

    for data in datas:
        cap_tokens = word_tokenize(data['caption'])
        for w in cap_tokens:
            word_counts[w] = word_counts.get(w, 0) + 1

    vocabs = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for w in word_counts:
        if word_counts[w] >= theshold:
            length = len(vocabs)
            vocabs[w] = length
    print('Bulid vocab size ', len(vocabs))
    
    d = json.dumps(vocabs)
    f = open(os.path.join(data_path, 'dict.json'), 'w')
    f.write(d)
    f.close()
    return vocabs

#data_path = '/data7/yll/dataset/MSCOCO/grid_feats/'
#data_path = '/data7/yll/dataset/flickr30k/grid_feats/'
#load_images(data_path)
#data_path = '/data2/cwj/distinctive_caption/caption_gen/general_caption/dataset/COCO_nongeneral/'
#data_path = '/data2/cwj/distinctive_caption/caption_gen/general_caption/dataset/COCO_general/'
#load_vocabs(data_path)