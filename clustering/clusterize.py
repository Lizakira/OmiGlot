import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import scanpy as sc
import anndata as ad
import numpy as np
from umap import UMAP
import json

def clusterize(modality_1_df, resolution=2.0, n_neighbors=15, n_comps=50, n_pcs=30):

    imputer = SimpleImputer(strategy="mean")

    modality_1_df = pd.DataFrame(
        imputer.fit_transform(modality_1_df),
        columns=modality_1_df.columns,
        index=modality_1_df.index
    )

    feature_names = np.array(modality_1_df.columns)
    print(feature_names)

    #We transpose the dataframe, so that features to be clustered become indexes
    X = modality_1_df.T

    X = X.values.astype(np.float32)


    adata = ad.AnnData(X=X)
    adata.obs_names = feature_names.astype(str)
    adata.var_names = modality_1_df.index.astype(str)

    sc.pp.scale(adata, zero_center=True, max_value=10)

    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack", random_state=42)

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=42)

    sc.tl.leiden(adata, resolution=resolution, key_added="leiden")

    sc.tl.umap(adata, random_state=42)

  
    cluster_labels = adata.obs["leiden"].astype(int).values  

    cluster_dict = {}

    for cluster_id in sorted(set(cluster_labels)):
        features_in_cluster = feature_names[cluster_labels == cluster_id]
        cluster_dict[f"cluster_{cluster_id}"] = features_in_cluster.tolist()

    print("Number of clusters:", len(cluster_dict))
    
    with open("clustering/data/clusters.json", "w") as f:
        json.dump(cluster_dict, f)
    
    print("Clusters saved in json file ")