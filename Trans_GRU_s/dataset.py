import json
import torch
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, images, vocabs, rev_vocabs, max_len, set_type):
        # set parameter
        self.images = images
        self.max_len = max_len
        self.vocabs = vocabs
        self.rev_vocabs = rev_vocabs
        self.set_type = set_type

        self.BOS = vocabs['<BOS>']
        self.EOS = vocabs['<EOS>']
        self.PAD = vocabs['<PAD>']
        self.UNK = vocabs['<UNK>']
        self.CLS = vocabs['<CLS>']

        # load data
        if self.set_type == 'P':
            self.load_multi_data(data_path)
        else:
            self.load_data(data_path)

    def load_data(self, data_path):
        self.datas = []
        dropdata = 0

        with open(data_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                jterm = json.loads(line.strip())
                if jterm['img_id'] in self.images and jterm['img_id']+'_2' in self.images:
                    self.datas.append(jterm)
                else:
                    dropdata += 1
        print('Total datas ', len(self.datas), 'drop ', dropdata, ' data')
    
    def load_multi_data(self, data_path):
        self.datas = []
        dropdata = 0
        with open(data_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                jterm = json.loads(line.strip())
                if jterm['img_id'] in self.images and jterm['img_id']+'_2' in self.images:
                    for s in jterm['sentences']:
                        sample = {'sentences': s, 'img_id': jterm['img_id']}
                        self.datas.append(sample)
                else:
                    dropdata += 1
        print('Total datas ', len(self.datas), 'drop ', dropdata, ' data')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # training and validation
        data = self.datas[index]
        img1 = torch.Tensor(self.images[data['img_id']])
        img2 = torch.Tensor(self.images[data['img_id']+'_2'])
        # print(img1.size())
        cap, cap_len = self.padding(data['sentences'].split(' '))
        return img1, img2, cap

    def get_data(self, index):
        # test mode
        data = self.datas[index]
        img1 = torch.Tensor(self.images[data['img_id']])
        img2 = torch.Tensor(self.images[data['img_id']+'_2'])
        ImgId = data['img_id']
        return ImgId, img1, img2, self.datas[index]['sentences']

    def padding(self, sent):
        if len(sent) > self.max_len-2:
            sent = sent[:self.max_len-2]
        text = list(map(lambda t: self.vocabs.get(t, self.UNK), sent))
        text = [self.BOS] + text + [self.EOS]
        length = len(text)
        # print('Length of text ', length)
        T = torch.cat([torch.LongTensor(text), torch.zeros(self.max_len - length).long()])
        return T, length

    def CLS(self):
        return self.CLS    