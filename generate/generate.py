import torch
from utils.utils import inverse_transform_y, transform_x


def generate(x_test, inference_artifacts):
    """
    Translate transcriptomic profiles into the target modality with a trained OmiGlot model.

    Args:
        x_test (pd.DataFrame or np.ndarray):
            Transcriptomic profiles of shape (B, N),
            where B = number of samples and N = number of source genes.

        inference_artifacts (dict):
            Dictionary containing all inference components:
                - "model":     trained OmiGlot model.
                - "x_imputer": imputer fitted on the training transcriptomic data.
                - "x_scaler":  scaler fitted on the training transcriptomic data.
                - "y_scaler":  scaler fitted on the training target data.
                - "device":    device used for inference ("cpu" or "cuda").

    Returns:
        dict:
            - "y_hat" (np.ndarray):
                Translated target profiles of shape (B, M),
                where M is the number of target features (original scale).

            - "s1" (np.ndarray):
                Target-side factor sigmoid(S1) of the importance matrix, of shape (M, K).

            - "s2" (np.ndarray):
                Cluster-side factor sigmoid(S2) of the importance matrix, of shape (K, |C|).

            - "s" (np.ndarray):
                Importance matrix S = sigmoid(S1) sigmoid(S2), of shape (M, |C|),
                giving the contribution of each gene cluster to each target feature.
    """
    model = inference_artifacts["model"]
    x_imputer = inference_artifacts["x_imputer"]
    x_scaler = inference_artifacts["x_scaler"]
    y_scaler = inference_artifacts["y_scaler"]
    device = inference_artifacts["device"]

    # Same preprocessing as for training (imputer and scaler fitted on the training set)
    x_test_processed = transform_x(x_test, x_imputer, x_scaler)
    x_test_tensor = torch.tensor(x_test_processed, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        y_hat_scaled, s1, s2, s = model(x_test_tensor)

    # Back to the original scale of the target modality
    y_hat = inverse_transform_y(y_hat_scaled.detach().cpu().numpy(), y_scaler)

    return {
        "y_hat": y_hat,
        "s1": s1.detach().cpu().numpy(),
        "s2": s2.detach().cpu().numpy(),
        "s": s.detach().cpu().numpy(),
    }


# Same function under the vocabulary of the paper
translate = generate