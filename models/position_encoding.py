"""
Various positional encodings for the transformer.
"""
import torch
from torch import nn

class SinusoidalPositionalEncoding(nn.Module): # not add dropout
    def __init__(self, d_model, max_len=5000):  
        super(SinusoidalPositionalEncoding, self).__init__()  
        # self.dropout = nn.Dropout(p=dropout)  
        
        # Precompute the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):  
        return self.pe[:x.size(0), :]

def build_position_encoding(args):
    # N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = SinusoidalPositionalEncoding(args.hidden_dim)
    elif args.position_embedding in ('v3', 'learned'):
        raise NotImplementedError
        # position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
