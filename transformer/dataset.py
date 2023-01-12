import json
import os
import torch
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, images, vocabs, rev_vocabs, max_len):
        # set parameter
        self.images = images
        self.max_len = max_len
        self.vocabs = vocabs
        self.rev_vocabs = rev_vocabs

        self.BOS = vocabs['<BOS>']
        self.EOS = vocabs['<EOS>']
        self.PAD = vocabs['<PAD>']
        self.UNK = vocabs['<UNK>']

        # load data
        self.load_data(data_path)

    def load_data(self, data_path):
        self.datas = []
        dropdata = 0
        raw_datas = json.load(open(data_path, 'r', encoding='utf-8'))
        for raw_data in raw_datas:
            if raw_data['img_id'] in self.images:
                self.datas.append(raw_data)
            else:
                dropdata += 1
        print('Total datas ', len(self.datas), 'drop ', dropdata, ' data')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # training and validation
        data = self.datas[index]
        img = torch.Tensor(self.images[data['img_id']])
        cap, cap_len = self.padding(data['caption'].split(' '))
        return img, cap

    def get_data(self, index):
        # test mode
        data = self.datas[index]
        img = torch.Tensor(self.images[data['img_id']])
        ImgId = data['img_id']
        cap = data['caption']
        return ImgId, img, cap

    def padding(self, sent):
        if len(sent) > self.max_len-3:
            sent = sent[:self.max_len-3]
        text = list(map(lambda t: self.vocabs.get(t, self.UNK), sent))
        text = [self.BOS] + text + [self.EOS]
        length = len(text)
        T = torch.cat([torch.LongTensor(text), torch.zeros(self.max_len - length).long()])
        return T, length