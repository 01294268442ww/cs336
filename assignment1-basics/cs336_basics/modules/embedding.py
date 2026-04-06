import torch
import torch.nn as nn

class Embedding(nn.Module):
    """
    Implement the Embedding class that inherits from torch.nn.Module and performs an embedding lookup. 
    Your implementation should follow the interface of PyTorch’s built-in nn.Embedding module.
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) -> None:
        super().__init__()

        """
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """

        self.W = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        nn.init.trunc_normal_(self.W, mean=0, std=1, a=-3, b=3)
    
    def forward(self, token_ids):

        # token_ids shape (B, L)
        B, L = token_ids.size()
        index = token_ids.reshape(-1)
        out = self.W.index_select(dim=0, index=index)
        out = out.reshape(B, L, -1)

        return out
