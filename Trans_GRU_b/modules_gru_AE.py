import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, n_embs, img_embs, n_hidden, vocab_size, dropout, max_len):
        super(Model, self).__init__()
        
        # initlize parameters
        self.n_embs = n_embs
        self.img_embs = img_embs
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dropout = dropout

        self.word_embedding = nn.Embedding(vocab_size, n_embs)
        self.img_project = nn.Sequential(nn.Linear(img_embs, n_hidden), nn.Sigmoid())
        self.position_encoding = PositionalEmb(n_hidden, dropout)
        self.img_encoder = Transformer(n_embs=n_embs, dim_ff=n_hidden, n_head=8, n_block=2, dropout=dropout)
        self.text_encoder = GRU_Encoder(n_embs=self.n_embs, dim_ff=self.n_hidden,dropout=dropout)
        self.decoder = GRU_Decoder(self.n_embs, self.n_embs, self.n_hidden, self.dropout)
        self.output_layer = nn.Linear(self.n_hidden, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(reduce=False, size_average=False, ignore_index=0)

    def encode_img(self, img1, img2):
        img1 = img1.view(img1.size(0), img1.size(1), -1)
        img2 = img2.view(img2.size(0), img2.size(1), -1)
        diff = img1 - img2
        Img = torch.cat((img1, img2, diff), dim=2)
        Img = Img.transpose(1, 2)
        Img = self.img_project(Img)
        Img = self.position_encoding(Img)
        Img = self.img_encoder(Img)
        G = torch.mean(Img, dim=1, keepdim=True)
        L = Img
        return G.squeeze(1), L

    def encode_text(self, Text):
        text_embs = self.word_embedding(Text)
        L, G = self.text_encoder(text_embs) # use LSTM last hidden state as global feature
        return G, L[:,:-1,:]

    def decode(self, Cap, G, L):
        text_embs = self.word_embedding(Cap)
        out = self.decoder(text_embs, G, L)
        out = self.output_layer(out)
        return out

    def forward(self, Img1, Img2, Cap):
        I, Imgs = self.encode_img(Img1, Img2)
        I = I.repeat(2,1,1)
        T, Text = self.encode_text(Cap)

        # print(I.size(),Imgs.size(),T.size(),Text.size())

        I_img = self.decode(Cap[:,:-1], I, Imgs)
        T_text = self.decode(Cap[:,:-1], T, Text)
        
        Cap = Cap.t()
        outs_img = I_img.transpose(0, 1)
        outs_text= T_text.transpose(0, 1)

        loss_img = self.criterion(outs_img.contiguous().view(-1, self.vocab_size), Cap[1:].contiguous().view(-1))
        loss_text = self.criterion(outs_text.contiguous().view(-1, self.vocab_size), Cap[1:].contiguous().view(-1))
        # print(torch.sum(loss_img), torch.sum(loss_text))
        loss = loss_img + loss_text
        return loss

    def generate(self, Img1, Img2, beam_size=1):
        I, Imgs = self.encode_img(Img1.unsqueeze(0), Img2.unsqueeze(0))
        I = I.repeat(2,1,1)
        Des = Variable(torch.ones(1, 1).long()).cuda()
        with torch.no_grad():
            for i in range(self.max_len):
                # embs = self.word_embedding(Des)
                out = self.decode(Des, I, Imgs)
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


class Transformer(nn.Module):
    def __init__(self, n_embs, dim_ff, n_head, dropout, n_block):
        super(Transformer, self).__init__()
        self.norm = LayerNorm(n_embs)
        self.layers = nn.ModuleList([AttnBlock(n_embs, dim_ff, n_head, dropout) for _ in range(n_block)])

    def forward(self, x):
        x = self.norm(x)
        for layer in self.layers:
            x = layer(x, x)
        return x

class AttnBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout):
        super(AttnBlock, self).__init__()
        self.attn = MultiHeadAttention(dim, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(dim, dropout) for _ in range(2)])

    def forward(self, x, m):
        x = self.sublayer[0](x, lambda x: self.attn(x, m, m))
        return self.sublayer[1](x, self.feed_forward)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_head, dropout):
        super(MultiHeadAttention, self).__init__()
        assert dim % n_head == 0
        self.d_k = dim // n_head
        self.n_head = n_head
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)]) 
        #其中0、1、2是Q、K、V输入的线性变换，3是最后连接输出的线性变换
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        weights = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(weights, dim=-1)

        if dropout is not None:
            p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_head * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, dim_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


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


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def subsequent_mask(batch, size):
    # mask out subsequent positions
    attn_shape = (batch, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class PositionalEmb(nn.Module):
    # Add special token for image1 and image 2 and img diff
    def __init__(self, n_hidden, dropout):
        super(PositionalEmb, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.img_embedding = nn.Embedding(3, n_hidden)

    def forward(self, x):
        # print('Images Embeddings ', x.size())
        batchSize = x.size(0)
        patchLen = x.size(1) // 3
        img1 = torch.LongTensor(batchSize, patchLen).fill_(0)
        img2 = torch.LongTensor(batchSize, patchLen).fill_(1)
        diff = torch.LongTensor(batchSize, patchLen).fill_(2)
        img_position = Variable(torch.cat((img1, img2, diff), dim=-1)).cuda()
        # print('Img Position ', img_position.size())
        img_embs = self.img_embedding(img_position)
        # print('Img Embedding ', img_embs.size())
        x = x + img_embs
        # print('X ', x.size())
        x = self.dropout(x)
        return x

class GRU_Encoder(nn.Module):
    def __init__(self, n_embs, dim_ff, dropout):
        super(GRU_Encoder, self).__init__()
        self.rnn = nn.GRU(
            input_size=n_embs,
            hidden_size=dim_ff,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

    def forward(self, x):
        # print('Encoder ', x.size())
        output, hidden = self.rnn(x,None)
        # print(output.size(),hidden.size())
        return output, hidden

class GRU_Decoder(nn.Module):
    def __init__(self, l_dims, n_embs, hidden_size, dropout):
        super(GRU_Decoder, self).__init__()
        self.input_size = n_embs
        self.hidden_size = hidden_size
        self.l_dims = l_dims
        self.dropout = dropout

        self.gru_l0 = nn.GRUCell(self.input_size, self.hidden_size, bias=False)
        self.gru_l1 = nn.GRUCell(self.hidden_size, self.hidden_size, bias=False)
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
        h_ = self.drop(h2)
        h_= self.attn(L, h_)
        h1 = self.linear(torch.cat([h[0], h_], dim=1)).tanh()
        #h = h1.unsqueeze(0) # [layers_num = 8, B, hidden_size]
        h = torch.cat([h1.unsqueeze(0),h2.unsqueeze(0)],dim=0,out=None) # [layers_num = 2, B, hidden_size]
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



