import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterTokenizer(nn.Module):
    """
    Encodes each gene cluster into one token embedding (OmiGlot tokenizer).

    Each cluster has its own small MLP, which maps the expression of the genes
    of that cluster to a d_model-dimensional token. The transcriptomic profile
    is thus represented as a sequence of |C| cluster tokens.

    Args:
        cluster_idx (list of array-like):
            List of gene clusters, where each array contains the indices of the
            genes belonging to a cluster. The length of the list defines the
            number of clusters |C|.
        d_model (int, optional):
            Dimension of the output token embeddings. Default: 128.
        hidden (int, optional):
            Hidden dimension of the MLP encoders. Default: 256.
        dropout (float, optional):
            Dropout probability applied inside each MLP encoder. Default: 0.2.

    Input:
        x_genes (torch.Tensor):
            Gene expression matrix of shape (B, N), where B is the batch size
            and N is the total number of genes.

    Output:
        torch.Tensor:
            Token matrix of shape (B, |C|, d_model). Layer normalization is
            applied along the d_model dimension.
    """

    def __init__(self, cluster_idx, d_model=128, hidden=256, dropout=0.2):

        super().__init__()

        self.cluster_idx = cluster_idx

        # One MLP encoder per gene cluster
        self.encoders = nn.ModuleList()

        for idx in cluster_idx:
            Gp = len(idx)                                 # number of genes in the cluster
            self.encoders.append(nn.Sequential(
                nn.Linear(Gp, hidden),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
                )
            )

        self.ln = nn.LayerNorm(d_model)

    def forward(self, x_genes):

        toks = []

        for p, idx in enumerate(self.cluster_idx):
            xp = x_genes[:, idx]          # (B, |c_p|) : genes of cluster p
            tp = self.encoders[p](xp)     # (B, d_model)
            toks.append(tp.unsqueeze(1))  # (B, 1, d_model)

        T = torch.cat(toks, dim=1)        # (B, |C|, d_model) : one token per cluster

        return self.ln(T)