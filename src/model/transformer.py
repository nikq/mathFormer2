import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learned) sinusoidal positional encoding.

    Returns a tensor of shape (1, t, d_model) given a position index tensor
    or uses the requested length `t` from the provided position tensor.
    """
    def __init__(self, block_size, d_model):
        super().__init__()
        pe = torch.zeros(block_size, d_model)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, block_size, d_model)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        # pos is a tensor like torch.arange(0, t).unsqueeze(0)
        t = pos.size(1)
        # return first t positions, moved to the same device as pos
        return self.pe[:, :t, :].to(pos.device)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        
        # Storage for visualization
        self.last_attn_weights = None

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # Store attention weights for visualization (detach to avoid graph retention)
        self.last_attn_weights = att.detach().cpu()

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.d_model, 4 * config.d_model)
        self.c_proj  = nn.Linear(4 * config.d_model, config.d_model)
        self.act     = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MathFormerConfig:
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4, d_model=128, pos_type='sinusoidal'):
        self.vocab_size = vocab_size # number of tokens in the vocabulary
        self.block_size = block_size # maximum length of input sequence
        self.n_layer = n_layer # number of transformer layers
        self.n_head = n_head # number of attention heads
        self.d_model = d_model # dimension of the model
        # pos_type: 'learned' or 'sinusoidal'
        self.pos_type = pos_type

class MathFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # choose positional encoding type
        if getattr(config, 'pos_type', 'learned') == 'learned':
            wpe = nn.Embedding(config.block_size, config.d_model)
        else:
            wpe = SinusoidalPositionalEncoding(config.block_size, config.d_model)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = wpe,
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.d_model),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Storage for visualization
        self.last_token_embeddings = None
        self.last_pos_embeddings = None

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        
        # Store for visualization
        self.last_token_embeddings = tok_emb.detach().cpu()
        self.last_pos_embeddings = pos_emb.detach().cpu()
        
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    def get_attention_weights(self):
        """
        Collects attention weights from all layers.
        Returns: List of tensors [Layer, Batch, Head, T, T]
        """
        weights = []
        for block in self.transformer.h:
            weights.append(block.attn.last_attn_weights)
        return weights

    def get_embeddings(self):
        """
        Returns the last forward pass embeddings.
        """
        return {
            "token": self.last_token_embeddings,
            "position": self.last_pos_embeddings
        }

    def get_all_weights(self):
        """
        Returns a dictionary of all model parameters as numpy arrays.
        """
        weights = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                weights[name] = param.detach().cpu().numpy().tolist()
        return weights
