import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def fit_preprocessors(
    x_train,
    y_train,
    x_impute_strategy="mean",
    y_impute_strategy="mean",
):
    """
    Fit les preprocessors sur train uniquement.
    """
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    x_imputer = SimpleImputer(strategy=x_impute_strategy)
    x_scaler = StandardScaler()

    y_imputer = SimpleImputer(strategy=y_impute_strategy)
    y_scaler = StandardScaler()

    x_train_imp = x_imputer.fit_transform(x_train)
    x_train_scaled = x_scaler.fit_transform(x_train_imp)

    y_train_imp = y_imputer.fit_transform(y_train)
    y_train_scaled = y_scaler.fit_transform(y_train_imp)

    return {
            "x_imputer": x_imputer,
            "x_scaler": x_scaler,
            "y_imputer": y_imputer,
            "y_scaler": y_scaler,
            "x_train_processed": x_train_scaled,
            "y_train_processed": y_train_scaled,
        }


def transform_x(x, x_imputer, x_scaler):
    x = np.asarray(x)
    x_imp = x_imputer.transform(x)
    x_scaled = x_scaler.transform(x_imp)
    return x_scaled


def transform_y(y, y_imputer, y_scaler):
    y = np.asarray(y)
    y_imp = y_imputer.transform(y)
    y_scaled = y_scaler.transform(y_imp)
    return y_scaled


def inverse_transform_y(y_scaled, y_scaler):
    return y_scaler.inverse_transform(y_scaled)


def build_cluster_indices(gene_columns, cluster_dict):

    """
    Build cluster-wise feature index mappings from source feature names.

    Args:
        gene_columns (list-like):
            Ordered list of source feature names.
            The position of each feature defines its index.

        cluster_dict (dict):
            Dictionary mapping cluster names to lists of feature names.
            Example:
                {
                    "cluster_1": ["geneA", "geneB"],
                    "cluster_2": ["geneC", "geneD"]
                }

    Returns:
        cluster_names (list):
            List containing cluster names.

        cluster_idx (list of np.ndarray):
            List of arrays, where each array contains the integer indices
            of features belonging to a cluster.
            Length of the list defines the number of clusters (C).
    """

    
    col_index = {g: i for i, g in enumerate(gene_columns)}
    cluster_names, cluster_idx = [], []

    for p, genes in cluster_dict.items():
        idx = [col_index[g] for g in genes if g in col_index]
    
        cluster_names.append(p)
        cluster_idx.append(np.array(idx, dtype=np.int64))
    return cluster_names, cluster_idx