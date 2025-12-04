import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def hardkuma_sample(alpha, beta, eps=1e-6, u_min=1e-4):
    u = torch.rand_like(alpha).clamp(u_min, 1.0 - u_min)
    log1m_u = torch.log1p(-u)
    t = torch.exp((1.0 / (beta + eps)) * log1m_u)
    one_minus_t = (1.0 - t).clamp(min=eps, max=1.0)
    x = torch.exp((1.0 / (alpha + eps)) * torch.log(one_minus_t))
    return x.clamp(eps, 1.0 - eps)


class Selector(nn.Module):
    def __init__(self, d_model, hidden=256):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
        self.out  = nn.Linear(hidden, 2)
        self.softplus = nn.Softplus()

    def forward(self, token_emb):
        h = F.gelu(self.proj(token_emb))
        alpha, beta = self.out(h).unbind(-1)
        alpha = (self.softplus(alpha) + 1.0).clamp(1.0, 10.0)
        beta  = (self.softplus(beta)  + 1.0).clamp(1.0, 10.0)
        return alpha, beta


class RationaleSelectorModel(nn.Module):
    def __init__(self, embedding_dim=None):
        super().__init__()
        self.selector = Selector(int(embedding_dim))

    def forward(self, embeddings, attention_mask=None):
        alpha, beta = self.selector(embeddings)
        return hardkuma_sample(alpha, beta)
