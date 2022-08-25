"""
B : batch size
L : sequence length (=number of points)
H : hidden dimension
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


class Encoder(nn.Module):
    def __init__(self, d_h, bsz, seq_len, n_layer, batch_first):
        super().__init__()
        self.lstm = nn.LSTM(input_size=d_h, hidden_size=d_h, num_layers=n_layer, batch_first=batch_first)
        self.batch_first = batch_first

        self.h0 = nn.Parameter(torch.randn(n_layer, bsz, d_h))
        self.c0 = nn.Parameter(torch.randn(n_layer, bsz, d_h))
        pass
    
    def forward(self, enc_inp):
        """
        h : (B, H, L)
        """
        if self.batch_first:
            enc_inp = enc_inp.permute(0, 2, 1)  # (B, L, H)
        else:
            enc_inp = enc_inp.permute(2, 0, 1)  # (L, B, H)

        return self.lstm(enc_inp, (self.h0, self.c0))

class Attention(nn.Module):
    def __init__(self, d_h):
        super().__init__()
        self.W_q = nn.Linear(d_h, d_h)
        self.W_ref = nn.Linear(d_h, d_h)
        self.vt = nn.Linear(d_h, 1)
        
    def forward(self, q, ref, mask=None):
        """
        Inputs:
            q : query (B, H)
            ref : encoder outputs (B, L, H) including end token
            mask : (B, L)
        Return:
            weights : (B, L)
        """
        Wq = self.W_q(q).unsqueeze(1)  # (B, 1 H)
        Wref = self.W_ref(ref)  # (B, L, H)
            
        u_i = self.vt(torch.tanh(Wref + Wq)).squeeze(-1)  # (B, L)
        if mask is not None:
            u_i.masked_fill_(mask, float('-1e9'))  # (B, L)
        return torch.log_softmax(u_i, dim=-1)
        # return u_i


class NETSP(nn.Module):
    def __init__(self, d_hidden, bsz, seq_len, n_layer, batch_first=True, **kwargs):
        super().__init__()
        self.d_h = d_hidden
        self.seq_len = seq_len

        self.start_token = nn.Parameter(torch.randn(bsz, d_hidden))

        self.h_dec0 = nn.Parameter(torch.randn(bsz, d_hidden))
        self.c_dec0 = nn.Parameter(torch.randn(bsz, d_hidden))

        self.embedding = nn.Conv1d(in_channels=2, out_channels=d_hidden, kernel_size=5, padding=2, padding_mode='circular')
        # self.embedding = nn.Sequential(
        #     Rearrange('b c n -> b n c'),
        #     nn.Linear(2, d_h),
        #     Rearrange('b n h -> b h n')
        # )
        self.encoder = Encoder(d_hidden, bsz, seq_len, n_layer, batch_first)
        self.decoder_lstm = nn.LSTMCell(input_size=d_hidden, hidden_size=d_hidden)
        self.attention = Attention(d_hidden)
    
    def forward(self, x, target):
        '''
        x : (B, 2, L)
        target : (B, L)
        size notations below are for batch_first=True
        '''
        toB = torch.arange(x.size(0))

        x_emb = self.embedding(x)  # (B, H, L)

        enc_out, enc_hid = self.encoder(x_emb)  # (B, L, H), (n_layers, B, H)
        
        dec_inp = self.start_token  # (B, H)
        h_t, c_t = self.h_dec0, self.c_dec0  # (B, H)
        mask = torch.zeros(x.size(0), x.size(-1), device=x.device).bool()  # (B, L)

        tour = []
        heatmap = []

        for t in range(self.seq_len):
            # TODO: teacher forcing?
            h_t, c_t = self.decoder_lstm(dec_inp, (h_t, c_t))  # (B, H)

            probs = self.attention(h_t, enc_out, mask)  # (B, L)
            # heatmap.append(probs.log().unsqueeze(-1))  # list of (B, L)
            heatmap.append(probs.unsqueeze(-1))  # list of (B, L)

            city = probs.argmax(dim=-1)  # (B)
            tour.append(city.unsqueeze(-1))  # list of (B, 1)

            # dec_inp = enc_out[toB, city, :]  # (B, H)
            dec_inp = enc_out[toB, target[:,t], :]  # (B, H) teacher forcing
            mask = mask.clone()
            mask[toB, city] = True
        
        tour = torch.cat(tour, dim=1)  # (B, L)
        heatmap = torch.cat(heatmap, dim=-1)  # (B, L, L)

        return tour, heatmap

if __name__ == '__main__':
    device = torch.device('cuda')
    bsz, seq_len, d_h = 100, 15, 128
    x = torch.rand(bsz, 2, seq_len).to(device)
    target = torch.randint(0,seq_len, (bsz, seq_len)).to(device)
    model = NETSP(d_h=d_h, bsz=bsz, seq_len=seq_len, n_layer=1).to(device)
    crit = nn.NLLLoss()
    # crit = nn.CrossEntropyLoss()
    oz = torch.optim.Adam(model.parameters(), lr=0.001)

    for e in range(1000):
        oz.zero_grad()
        tour, heatmap = model(x, target)
        loss = crit(heatmap, target)
        loss.backward()
        oz.step()
        if e % 100 == 0:
            print(loss.mean().item())
        

    # tour, heatmap = model(x)
    # from global_utils import compute_tour_length
    # print(compute_tour_length(x.permute(0,2,1), tour)[0])
    # print(tour[0])
    # print(tour.shape)
    # print(heatmap.shape)