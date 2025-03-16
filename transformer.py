import torch
from torch import nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from functools import reduce
from torch import Tensor
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self,d_model=512, num_heads=8):
        super(EncoderLayer,self).__init__()
        self.attn = nn.MultiheadAttention(d_model,num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model,4 * d_model),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self,x):
        attn_output,_ = self.attn(x,x,x)
        x = self.norm(attn_output + x)
        
        ffn_output = self.ffn(x)
        x = self.norm2(ffn_output + x)
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self,d_model=512,num_heads=8,num_layers=6):
        super(TransformerEncoder,self).__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model,num_heads) for _ in range(num_layers)
        ])
    
    def forward(self,x):
        return reduce(lambda acc, cur: cur(acc),self.encoder_layers,x)
    
class DecoderLayer(nn.Module):
    def __init__(self,d_model=512
                 ,num_heads=8):
        super(DecoderLayer,self).__init__()
        self.masked_attn = nn.MultiheadAttention(d_model,num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model,num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model,4 * d_model),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self,x,enc_output):
        # masked multihead attention
        masked_attn_output,_ = self.masked_attn(x,x,x,attn_mask=self.generate_causal_mask(x.shape[0]))
        x = self.norm1(masked_attn_output + x)
        # multihead attention
        attn_output,_ = self.attn(x,enc_output,enc_output)
        x = self.norm2(attn_output + x)
        # ffn
        ffn_output = self.ffn(x)
        x = self.norm3(ffn_output + x)
        
        return x
        
    def generate_causal_mask(self,seq_len):
        mask = torch.ones(seq_len,seq_len)
        mask = torch.triu(mask,diagonal=1)
        mask = mask.masked_fill(mask == 1,float('-inf'))
        return mask
        
class TransformerDecoder(nn.Module):
    def __init__(self,d_model=512
                 ,num_heads=8,num_layers=6):
    # 6 decoder layers consists of 3 sublayers: 
    # masked multihead attention
    # multihead attention
    # ffn#
        super(TransformerDecoder,self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model,num_heads) for _ in range(num_layers)
        ])
    def forward(self,x,enc_output):
        for layer in self.layers:
            x = layer(x,enc_output)
        return x
    

class Transformer(nn.Module):
    '''Implementation of the transformer'''
    def __init__(self,d_model=512,num_heads=8,num_layers=6,vocab_size=10000):
        super(Transformer,self).__init__()
        
        self.enc_blk = TransformerEncoder(d_model,num_heads,num_layers)
        
        self.dec_blk = TransformerDecoder(d_model,num_heads,num_layers)
        
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.embedding = nn.Embedding(vocab_size,d_model)
        
        self.linear = nn.Linear(d_model,vocab_size)
        
    def forward(self,src,tgt,src_mask=None,tgt_mask=None):
        src_emb = self.embedding(src)
        src_emb_pos = self.positional_encoding(src_emb)
        enc_output = self.enc_blk(src_emb_pos)
        
        tgt_emb = self.embedding(tgt)
        tgt_emb_pos = self.positional_encoding(tgt_emb)
        dec_output = self.dec_blk(tgt_emb_pos,enc_output)
        
        logits = self.linear(dec_output)
        
        return logits
        
if __name__ == '__main__':
    seq_len = 10
    mask = torch.ones(seq_len,seq_len)
    mask = torch.triu(mask,diagonal=1)
    mask = mask.masked_fill(mask == 1,float('-inf'))
    print(mask)