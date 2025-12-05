import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def hardkuma_sample(alpha, beta, eps=1e-6, u_min=1e-4):
    """
    Reparameterised sample from Kumaraswamy(alpha, beta) used for hard gates.
    Uses the closed-form inverse CDF: x = (1 - (1 - u)^(1/beta))^(1/alpha).
    """
    u = torch.rand_like(alpha).clamp(u_min, 1.0 - u_min)
    inv_alpha = 1.0 / (alpha + eps)
    inv_beta = 1.0 / (beta + eps)
    inner = 1.0 - torch.pow(1.0 - u, inv_beta)
    x = torch.pow(inner.clamp(min=eps, max=1.0), inv_alpha)
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

    def forward(self, embeddings, attention_mask=None, hard=True):
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        alpha, beta = self.selector(embeddings)  # [B, L], [B, L]
        z = hardkuma_sample(alpha, beta)         # soft ∈ (0,1)

        if hard:
            # hard gate in forward, soft grad in backward
            h = (z > 0.5).float()
            gates = h + (z - z.detach())
        else:
            # pure soft gates
            gates = z

        if attention_mask is not None:
            gates = gates * attention_mask.float()

        return gates
