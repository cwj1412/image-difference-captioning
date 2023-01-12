import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, n_embs, img_embs, n_hidden, vocab_size, dropout, max_len=100):
        super(Model, self).__init__()
        
        # initlize parameters
        self.n_embs = n_embs
        self.img_embs = img_embs
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dropout = dropout

        # Text
        self.word_embedding = WordEmbeddings(self.n_embs, self.vocab_size)
        self.encoder = GRU_Encoder(img_embs=img_embs, n_embs=n_embs, dim_ff=n_hidden,dropout=dropout)
        # Images
        self.decoder = GRU_Decoder(self.n_embs, self.n_embs, self.n_hidden, self.dropout)
        self.output_layer = nn.Linear(self.n_hidden, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(reduce=False, size_average=False, ignore_index=0)

    def encode_img(self, img1, img2):
        #Img = torch.cat((img1, img2,img1-img2), dim=2) #[50,2048,49*3]
        #Img = img1 - img2
        #Img = img1 + img2
        #Img = torch.mul(img1,img2)
        
        img1 = img1.view(img1.size(0), img1.size(1), -1)
        img2 = img2.view(img2.size(0), img2.size(1), -1)
        Img = torch.cat((img1, img2), dim=2)
        Img = Img.transpose(1,2)

        # [50,2048,49*3] -> [50,49*3,2048]
        # print('Concat images ', Img.size())
        L, G = self.encoder(Img)
        # print('H ', H.size())
        return G, L[:,:-1,:]

    def decode(self, Des, G, L):
        embs = self.word_embedding(Des)
        out = self.decoder(embs, G, L)
        out = self.output_layer(out)
        return out

    def forward(self, Img1, Img2, Des):
        # print('Des ', Des.size())
        Des_emb = self.word_embedding(Des)
        # print('Embs ', Des_emb.size())
        G, L = self.encode_img(Img1, Img2)
        outs = self.decode(Des[:,:-1], G, L)
        Des = Des.t()
        outs = outs.transpose(0, 1)
        # print(outs.size(),Des.size())
        loss = self.criterion(outs.contiguous().view(-1, self.vocab_size), Des[1:].contiguous().view(-1))
        return loss

    def generate(self, img1, img2, beam_size = 1):
        G, L = self.encode_img(img1.unsqueeze(0), img2.unsqueeze(0))
        Des = Variable(torch.ones(1, 1).long()).cuda()
        with torch.no_grad():
            for i in range(self.max_len):
                out = self.decode(Des, G, L)
                prob = out[:, -1]
                _, next_w = torch.max(prob, dim=-1, keepdim=True)
                next_w = next_w.data
                Des = torch.cat([Des, next_w], dim=-1)
        return Des[:, 1:] 


class WordEmbeddings(nn.Module):
    def __init__(self, n_emb, vocabs):
        super(WordEmbeddings, self).__init__()
        self.n_emb = n_emb
        self.word2vec = nn.Embedding(vocabs, n_emb)

    def forward(self, x):
        return self.word2vec(x)


class GRU_Encoder(nn.Module):
    def __init__(self, img_embs, n_embs, dim_ff, dropout):
        super(GRU_Encoder, self).__init__()
        self.rnn = nn.GRU(
            input_size=n_embs,
            hidden_size=dim_ff,
            num_layers=8,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.linear = nn.Linear(img_embs, n_embs)
        self.norm = LayerNorm(n_embs)

    def forward(self, x):
        # print('Encoder ', x.size())
        x = self.linear(x) # [50*49*img_embs] -> [50*49*n_embs]
        x = self.norm(x)
        output, hidden = self.rnn(x,None)
        # print(hidden)
        return output, hidden

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class GRU_Decoder(nn.Module):
    def __init__(self, l_dims, n_embs, hidden_size, dropout):
        super(GRU_Decoder, self).__init__()
        self.input_size = n_embs
        self.hidden_size = hidden_size
        self.l_dims = l_dims
        self.dropout = dropout

        self.gru_l0 = nn.GRUCell(self.input_size, self.hidden_size, bias=False)
        self.gru_l1 = nn.GRUCell(self.hidden_size, self.hidden_size, bias=False)
        self.gru_l2 = nn.GRUCell(self.hidden_size, self.hidden_size, bias=False)
        self.gru_l3 = nn.GRUCell(self.hidden_size, self.hidden_size, bias=False)
        self.gru_l4 = nn.GRUCell(self.hidden_size, self.hidden_size, bias=False)
        self.gru_l5 = nn.GRUCell(self.hidden_size, self.hidden_size, bias=False)
        self.gru_l6 = nn.GRUCell(self.hidden_size, self.hidden_size, bias=False)
        self.gru_l7 = nn.GRUCell(self.hidden_size, self.hidden_size, bias=False)
        self.linear = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        self.norm = LayerNorm(self.hidden_size)
        self.drop = nn.Dropout(self.dropout)

    def attn(self, l, h):
        """
        args:
            l: additional local information [B, N, l_dims = hidden_size]
            h: the current time step hidden state [B, hidden_size]

        return:
            h_: this time step context [B, hidden_size]
        """
        h = h.unsqueeze(1)
        weights = torch.bmm(h, l.transpose(1,2))
        attn_weights = F.softmax(weights, dim=-1)
        h_ = torch.bmm(attn_weights, l)
        return h.squeeze(1)

    def step_decoding(self, x, L, h):
        #print(x.size(),h.size())
        h1 = self.gru_l0(x, h[0])
        h2 = self.gru_l1(h1, h[1])
        h3 = self.gru_l2(h2, h[2])
        h4 = self.gru_l3(h3, h[3])
        h5 = self.gru_l4(h4, h[4])
        h6 = self.gru_l5(h5, h[5])
        h7 = self.gru_l6(h6, h[6])
        h8 = self.gru_l7(h7, h[7])
        h_ = self.drop(h8)
        h_= self.attn(L, h_)
        h1 = self.linear(torch.cat([h[0], h_], dim=1)).tanh()
        h = torch.cat([h1.unsqueeze(0),h2.unsqueeze(0),h3.unsqueeze(0),h4.unsqueeze(0),h5.unsqueeze(0),h6.unsqueeze(0),h7.unsqueeze(0),h8.unsqueeze(0)],dim=0,out=None) # [layers_num = 8, B, hidden_size]
        #h1 = self.drop(h1)
        return h

    def forward(self, X, G, L):
        output = []
        h = G
        for t in range(X.size(1)):
            # one step decoding
            x = X[:, t, :]
            h = self.step_decoding(x, L, h)
            output.append(h[0])
        output = torch.stack(output, dim=1)  # [B, MAX_LEN, hidden_size]
        
        #print(output)
        return self.norm(output)


