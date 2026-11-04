import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def train_model(
    model,
    x_train_tensor,
    y_train_tensor,
    num_epochs=150,
    batch_size=8,
    lr=1e-3,
    device=None,
    verbose=True
):
    """
    Train an OmiGlot model for cross-omic translation.

    The model is trained with the Adam optimizer (weight decay 1e-4) and the
    Smooth L1 loss between translated and measured target profiles.

    Args:
        model (torch.nn.Module):
            OmiGlot model to train.

        x_train_tensor (torch.Tensor):
            Transcriptomic profiles of shape (B, N),
            where B = number of training samples and N = number of source genes.

        y_train_tensor (torch.Tensor):
            Target profiles of shape (B, M),
            where M = number of target features.

        num_epochs (int, optional):
            Number of training epochs. Default: 150.

        batch_size (int, optional):
            Number of samples per mini-batch. Default: 8.

        lr (float, optional):
            Learning rate of the Adam optimizer. Default: 1e-3.

        device (str, optional):
            Device used for training ("cpu" or "cuda"). If None, CUDA is used
            when available. Default: None.

        verbose (bool, optional):
            Whether to print the training loss at each epoch. Default: True.

    Returns:
        model (torch.nn.Module):
            Trained OmiGlot model.

        history (dict):
            Training history:
                - "loss": list of epoch-wise training losses.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    x_train_tensor = x_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)

    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"loss": []}

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        n_samples = 0

        for xb, yb in loader:
            optimizer.zero_grad()

            y_hat, s1, s2, s = model(xb)                      # translated profiles + importance matrices

            loss = F.smooth_l1_loss(y_hat, yb, beta=1.0)      # Smooth L1 loss (beta = 1)

            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

        epoch_loss = total_loss / n_samples
        history["loss"].append(epoch_loss)

        if verbose:
            print(f"Epoch {epoch + 1:03d} | loss={epoch_loss:.4f}")

    return model, history