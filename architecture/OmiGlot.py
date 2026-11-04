import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.ClusterTokenizer import ClusterTokenizer


class OmiGlot(nn.Module):
    """
    OmiGlot: interpretable transformer-based framework for cross-omic translation.

    The source genes are grouped into clusters, each cluster is encoded as a token,
    and a Transformer encoder models interactions between gene clusters. Each target
    feature then aggregates the cluster representations through a low-rank importance
    matrix S = sigmoid(S1) sigmoid(S2), which quantifies the contribution of each gene
    cluster to each target feature.

    Args:
        num_features (int):
            Total number of source features (genes).

        cluster_idx (list of array-like):
            List of gene clusters, where each array contains the indices
            of the genes belonging to a cluster.
            The length of the list defines the number of clusters |C|.

        m_outputs (int):
            Number of target features M (e.g. metabolites, proteins, miRNAs).

        k_latent (int):
            Rank K of the low-rank factorization of the importance matrix S.

        d_model (int, optional):
            Dimension of token embeddings and Transformer representations.
            Default: 128.

        hidden_dim (int, optional):
            Hidden dimension used inside the cluster tokenizer.
            Default: 256.

        nhead (int, optional):
            Number of attention heads in the Transformer encoder.
            Default: 4.

        num_layers (int, optional):
            Number of Transformer encoder layers.
            Default: 2.

        dim_feedforward (int, optional):
            Dimension of the feedforward network inside Transformer layers.
            Default: 128.

        dropout (float, optional):
            Dropout probability applied throughout the model.
            Default: 0.3.

    Input:
        x (torch.Tensor):
            Transcriptomic profiles of shape (B, N),
            where B = batch size and N = number of source genes.

    Output:
        y_hat (torch.Tensor):
            Translated target profiles of shape (B, M).

        s1 (torch.Tensor):
            Target-side factor sigmoid(S1), of shape (M, K).

        s2 (torch.Tensor):
            Cluster-side factor sigmoid(S2), of shape (K, |C|).

        s (torch.Tensor):
            Importance matrix S = sigmoid(S1) sigmoid(S2), of shape (M, |C|),
            giving the contribution of each gene cluster to each target feature.
    """

    def __init__(
        self,
        num_features: int,
        cluster_idx,
        m_outputs: int,
        k_latent: int,
        d_model: int = 128,
        hidden_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()

        self.cluster_idx = cluster_idx
        self.m_outputs = m_outputs
        self.k_latent = k_latent
        self.d_model = d_model

        self.tokenizer = ClusterTokenizer(
            cluster_idx=cluster_idx,
            d_model=d_model,
            hidden=hidden_dim,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Factors S1 (M x K) and S2 (K x |C|) of the importance matrix, before the sigmoid.
        # The parameter names A1 / A2 are kept so that previously saved checkpoints still load.
        self.A1 = nn.Parameter(torch.randn(m_outputs, k_latent) * 0.5)
        self.A2 = nn.Parameter(torch.randn(k_latent, len(cluster_idx)) * 0.5)

        self.out_weight = nn.Parameter(torch.randn(m_outputs, d_model) * (1.0 / d_model**0.5))
        self.out_bias = nn.Parameter(torch.zeros(m_outputs))

    def forward(self, x: torch.Tensor):

        tokens = self.tokenizer(x)               # (B, |C|, d_model) : one token per gene cluster

        h = self.transformer(tokens)             # (B, |C|, d_model) : contextualized cluster representations

        s1 = torch.sigmoid(self.A1)              # (M, K)
        s2 = torch.sigmoid(self.A2)              # (K, |C|)

        s = torch.matmul(s1, s2)                 # (M, |C|) : importance matrix S

        o = torch.einsum('mp,npd->nmd', s, h)    # (B, M, d_model) : aggregated target representations

        y_hat = (o * self.out_weight.unsqueeze(0)).sum(dim=-1) + self.out_bias   # (B, M)

        return y_hat, s1, s2, s


# Former name of the model (CIKM version), kept so that existing code and notebooks keep working.
MultiTabGen = OmiGlot
