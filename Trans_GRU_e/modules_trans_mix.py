import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, n_embs, img_embs, n_hidden, vocab_size, dropout, max_len, CLS):
        super(Model, self).__init__()
        
        # initlize parameters
        self.n_embs = n_embs
        self.img_embs = img_embs
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dropout = dropout
        self.CLS = CLS

        # Text
        self.word_embedding = WordEmbeddings(self.n_embs, self.vocab_size)
        self.position_embedding = ImgPositionEncoding(n_hidden, dropout, 4)
        self.img_project = nn.Sequential(nn.Linear(img_embs, n_hidden), nn.Sigmoid())
        self.img_encoder = Transformer(n_embs=n_embs, dim_ff=n_hidden, n_head=8, n_block=3, dropout=dropout)
        self.text_encoder = Transformer(n_embs=n_embs, dim_ff=n_hidden, n_head=8, n_block=4, dropout=dropout)
        self.decoder = TransformerDecoder(dim=self.n_hidden, dim_ff=self.n_hidden, n_head=8, n_block=1, dropout=dropout)

        self.output_layer = nn.Linear(self.n_hidden, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(reduce=False, size_average=False, ignore_index=0)
    '''
    def encode_img(self, img1, img2):
        img1 = img1.view(img1.size(0), img1.size(1), -1)
        img2 = img2.view(img2.size(0), img2.size(1), -1)

        #Img = img1 - img2
        #Img = torch.cat((img1, img2), dim=2)
        diff = img1 - img2
        Img = torch.cat((img1, img2, diff), dim=2)

        Img = Img.transpose(1, 2)
        Img = self.img_project(Img)
        Img = self.position_embedding(Img)

        L = self.img_encoder(Img)
        G = torch.mean(L, dim=1, keepdim=True)
        return G, L

    def encode_text(self, Text):
        text_embs = self.word_embedding(Text)
        L = self.text_encoder(text_embs)
        G = torch.mean(L, dim=1, keepdim=True)
        return G, L  
    '''

    def encode_img(self, img1, img2):
        img1 = img1.view(img1.size(0), img1.size(1), -1)
        img2 = img2.view(img2.size(0), img2.size(1), -1)
        diff = img1 - img2
        Img = torch.cat((img1, img2, diff), dim=2)
        Img = Img.transpose(1, 2)
        Img = self.img_project(Img)
        CLS = torch.tensor([[self.CLS]])
        CLS = CLS.repeat(Img.size(0), 1)
        CLS = self.word_embedding(CLS.cuda())
        Img = torch.cat((CLS, Img), dim=1)
        Img = self.position_embedding(Img)
        Img = self.img_encoder(Img)
        G = Img[:, 0, :].unsqueeze(1)
        L = Img[:, 1:, :]
        return G, L

    def encode_text(self, Text):
        batch_size = Text.size(0)
        text_embs = self.word_embedding(Text)
        CLS = torch.mean(text_embs, dim=1, keepdim=True)
        #CLS = torch.LongTensor(batch_size, 1).fill_(self.CLS).cuda()
        #CLS = self.word_embedding(CLS)
        text_embs = torch.cat((CLS, text_embs), dim=1)
        T = self.text_encoder(text_embs)
        CLS_T = T[:, 0, :].unsqueeze(1)
        return CLS_T, T[:, 1:, :]
    
    def decode(self, Des, G, L, mask):
        embs = self.word_embedding(Des)
        M = torch.cat((G, L),dim=1)
        out = self.decoder(embs, M, mask)
        out = self.output_layer(out)
        return out

    def forward(self, Img1, Img2, Des):
        Des_emb = self.word_embedding(Des)
        I, Imgs = self.encode_img(Img1, Img2)
        T, Text = self.encode_text(Des)
        mask = Variable(subsequent_mask(Des.size(0), Des.size(1)-1), requires_grad=False).cuda()

        I_img = self.decode(Des[:,:-1], I, Imgs, mask=mask)
        #I_text = self.decode(Des[:,:-1], I, Text, mask=mask)
        T_img = self.decode(Des[:,:-1], T, Imgs, mask=mask)

        Des = Des.t()
        outs_img = I_img.transpose(0, 1)
        #outs_text = I_text.transpose(0, 1)
        outs_text = T_img.transpose(0, 1)
        loss_img = self.criterion(outs_img.contiguous().view(-1, self.vocab_size), Des[1:].contiguous().view(-1))
        loss_text = self.criterion(outs_text.contiguous().view(-1, self.vocab_size), Des[1:].contiguous().view(-1))
        loss = loss_img + loss_text
        return loss

    def generate(self, img1, img2, beam_size = 1):
        # Img_G, Imgs = self.encode_img(img1.unsqueeze(0), img2.unsqueeze(0))
        I, Imgs = self.encode_img(img1.unsqueeze(0), img2.unsqueeze(0))
        Des = Variable(torch.ones(1, 1).long()).cuda()
    
        Des = self.beam_search(I, Imgs, beam_size)
        return Des.squeeze()

    def greedy(self, Imgs):
        Des = Variable(torch.ones(1, 1).long()).cuda()
        with torch.no_grad():
            for i in range(self.max_len):
                out = self.decode(Des, Imgs, Variable(subsequent_mask(Des.size(0), Des.size(1))).cuda())
                prob = out[:, -1]
                _, next_w = torch.max(prob, dim=-1, keepdim=True)
                next_w = next_w.data
                Des = torch.cat([Des, next_w], dim=-1)
        return Des[:, 1:]

    def beam_search(self, I, Imgs, beam_size):
        LENGTH_NORM = True
        batch_size = Imgs.size(0)
        startTokenArray = Variable(torch.ones(batch_size, 1).long()).cuda()
        backVector = torch.LongTensor(beam_size).cuda()
        torch.arange(0, beam_size, out=backVector)
        backVector = backVector.unsqueeze(0).repeat(batch_size, 1)
        backVector = Variable(backVector)

        tokenArange = torch.LongTensor(self.vocab_size).cuda()
        torch.arange(0, self.vocab_size, out=tokenArange)
        tokenArange = Variable(tokenArange)

        beamTokensTable = torch.LongTensor(batch_size, beam_size, self.max_len).fill_(2) # end Token
        beamTokensTable = Variable(beamTokensTable.cuda())
        backIndices = torch.LongTensor(batch_size, beam_size, self.max_len).fill_(-1)
        backIndices = Variable(backIndices.cuda())

        aliveVector = beamTokensTable[:, :, 0].eq(2).unsqueeze(2)

        for i in range(self.max_len-1):
            if i == 0:
                # start Token
                Des = startTokenArray
                out = self.decode(Des, I, Imgs, Variable(subsequent_mask(Des.size(0), Des.size(1))).cuda())
                probs = out[:, -1]
                topProbs, topIdx = probs.topk(beam_size, dim=1)
                beamTokensTable[:, :, 0] = topIdx.data
                ProbSums = topProbs
            else:
                # print('\n')
                Des = beamTokensTable[:, :, :i].squeeze(0)
                # print('Des tokens', Des.size())
                out = self.decode(Des, I, Imgs.repeat(beam_size, 1, 1), Variable(subsequent_mask(Des.size(0), Des.size(1))).cuda())
                # print('Out ', out.size())
                # probCurrent = out[:, -1].view(batch_size, beam_size, self.vocab_size)
                # probCurrent = F.log_softmax(out[:, -1,:].view(batch_size, beam_size, self.vocab_size), dim=-1)
                probCurrent = out[:, -1,:].view(batch_size, beam_size, self.vocab_size)
                if LENGTH_NORM:
                    probs = probCurrent * (aliveVector.float() / (i+1))
                    # print(aliveVector)
                    # print(aliveVector.float() / (i+1))
                    coeff_ = aliveVector.eq(0).float() + (aliveVector.float() * i / (i+1))
                    probs += ProbSums.unsqueeze(2) * coeff_
                else:
                    probs = probCurrent * (aliveVector.float())
                    probs += ProbSums.unsqueeze(2)
                
                mask_ = aliveVector.eq(0).repeat(1, 1, self.vocab_size)
                mask_[:, :, 0] = 0 # Zeros out all except first row for ended beams
                minus_infinity_ = torch.min(probs).item()

                probs.data.masked_fill_(mask_.data, minus_infinity_)
                probs = probs.view(batch_size, -1)

                tokensArray = tokenArange.unsqueeze(0).unsqueeze(0).repeat(batch_size, beam_size, 1)
                tokensArray.masked_fill_(aliveVector.eq(0), 2)
                tokensArray = tokensArray.view(batch_size, -1)
                backIndexArray = backVector.unsqueeze(2).repeat(1, 1, self.vocab_size).view(batch_size, -1)

                topProbs, topIdx = probs.topk(beam_size, dim=1)
                ProbSums = topProbs
                beamTokensTable[:, :, i] = tokensArray.gather(1, topIdx)
                backIndices[:, :, i] = backIndexArray.gather(1, topIdx)

            aliveVector = beamTokensTable[:, :, i:i + 1].ne(2)
            aliveBeams = aliveVector.data.long().sum()
            finalLen = i
            if aliveBeams == 0:
                break

        beamTokensTable = beamTokensTable.data
        backIndices = backIndices.data

        RECOVER_TOP_BEAM_ONLY = False
        tokenIdx = finalLen
        backID = backIndices[:, :, tokenIdx]
        tokens = []
        while tokenIdx >= 0:
            tokens.append(beamTokensTable[:, :, tokenIdx].gather(1, backID).unsqueeze(2))
            backID = backIndices[:, :, tokenIdx].gather(1, backID)
            tokenIdx = tokenIdx - 1

        tokens.append(startTokenArray.unsqueeze(2).repeat(1, beam_size, 1).data)
        tokens.reverse()
        tokens = torch.cat(tokens, 2)
        seqLen = tokens.ne(2).long().sum(dim=2)

        if RECOVER_TOP_BEAM_ONLY:
            tokens = tokens[:, 0]
            seqLen = seqLen[:, 0]
            
        return Variable(tokens)

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.Wq = nn.Linear(self.dim, self.dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V):
        Q = self.Wq(Q)
        weights = torch.bmm(Q, K.transpose(-2, -1))
        weights = self.softmax(weights)
        attn_out = torch.bmm(weights, V)
        return attn_out.squeeze(1)

class WordEmbeddings(nn.Module):
    def __init__(self, n_embs, vocabs):
        super(WordEmbeddings, self).__init__()
        self.n_embs = n_embs
        self.word2vec = nn.Embedding(vocabs, n_embs)

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
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        # print('Query ', query.size())
        # print('Key ', key.size())
        # print('Value ', value.size())
        weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # print('Weights ', weights.size())
        if mask is not None:
            # print('Weights ', weights.size())
            # print('Mask weights ', mask.size())
            weights = weights.masked_fill(mask == 0, -1e9)
            #print('Weights ', weights)
        p_attn = F.softmax(weights, dim=-1)
        if dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
            # print('Transformer Mask ', mask.size())
        nbatches = query.size(0)
        # print('Query ', query.size())
        # print('Key ', key.size())
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
    mask = torch.from_numpy(subsequent_mask) == 0
    # return mask[:,1:,:]
    return mask

class TransformerDecoder(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout, n_block):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(dim, dim_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(dim)

    def forward(self, x, v, mask):
        for layer in self.layers:
            x = layer(x, v, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(dim, n_head, dropout)
        self.attn = MultiHeadAttention(dim, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(dim, dropout) for _ in range(3)])

    def forward(self, x, v, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, lambda x: self.attn(x, v, v))
        return self.sublayer[2](x, self.feed_forward)

class ImgPositionEncoding(nn.Module):
    # Add special token for global and image1 and image 2 and img diff
    def __init__(self, n_hidden, dropout, max_len):
        super(ImgPositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.nn.Embedding(max_len, n_hidden)

    def forward(self, x):
        batchSize = x.size(0)
        patchLen = (x.size(1)-1) // 3
        CLS = torch.LongTensor(batchSize, 1).fill_(0)
        img1 = torch.LongTensor(batchSize, patchLen).fill_(1)
        img2 = torch.LongTensor(batchSize, patchLen).fill_(2)
        diff = torch.LongTensor(batchSize, patchLen).fill_(3)
        img_position = Variable(torch.cat((CLS, img1, img2, diff), dim=-1)).cuda()
        pe_embs = self.pe(img_position)
        x = x + pe_embs
        x = self.dropout(x)
        return x