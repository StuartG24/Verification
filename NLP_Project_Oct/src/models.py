# Prepare Training etc Data
#

import numpy as np
from sklearn.model_selection import train_test_split

def prep_data_splits(queries_df):
    """
    Prepare X,y with Train, Val, Test
    !! Assumes no K-Fold validation, and need to split before PCA

    Args:
        Queries_df, sentences and embeddings: Pandas df

    Returns:
        X_train, y_train etc: np ndarray
    """

    # Select features, X and target y
    X = np.vstack(queries_df['query-embedding'].values)
    y = queries_df['query-is-non-medical'].to_numpy().astype(np.float32)  # Convert bool to float32

    # Ref for later mapping back to original query test, ie before the splits
    ref_ids = queries_df['ref-id'].values

    # Train, validate, test split 60:20:20
    # NB: Classes are possibly imbalanced so use stratification
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(X, y, ref_ids, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(X_temp, y_temp, ids_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Full queries df shape: {queries_df.shape}")
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    print(f"Ref ids: {ref_ids.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, ids_train, ids_val, ids_test

# PCA on embeddings
#

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def pca_create(X_train, X_val, X_test, components=30):

    # TODO: components=30 reduces to 41% explained variance ... but better performing model

    # Repeatable Pipeline, with scaling
    pipe = Pipeline([
        ("center", StandardScaler(with_mean=True, with_std=False)),
        # ("pca", PCA(n_components=0.75, svd_solver="full", random_state=42)),
        ("pca", PCA(n_components=components, svd_solver="full", random_state=42)), 
    ])

    # Fit to X_train and transform train, val, test
    train_PCA_pipe = pipe.fit(X_train)
    X_train_pca = train_PCA_pipe.transform(X_train)
    X_val_pca = train_PCA_pipe.transform(X_val)
    X_test_pca = train_PCA_pipe.transform(X_test)
    # X_train_pca = pipe.fit_transform(X_train)
    # X_val_pca = pipe.transform(X_val)
    # X_test_pca = pipe.transform(X_test)

    print('Dimensions after PCA. Train, val, test')
    print("Original", X_train.shape[1], X_val.shape[1], X_test.shape[1])
    print("Reduced", X_train_pca.shape[1], X_val_pca.shape[1], X_test_pca.shape[1])
    print(f'Explained variance total: {pipe.named_steps["pca"].explained_variance_ratio_.sum():.4f}')

    cum_var = np.cumsum(pipe.named_steps["pca"].explained_variance_ratio_)
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
    plt.axhline(0.90, color='red', linestyle='--', linewidth=1)  # 90% threshold
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('Cumulative explained variance by PCA components')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Save pipeline for later use with the NN model
    # import joblib
    # joblib.dump(pipe, "pca_embeddings_pipeline.joblib")
    # Reload with NN model
    # pipe = joblib.load("pca_embeddings_pipeline.joblib")
    # X_new_pca = pipe.transform(X_new)
    
    return train_PCA_pipe, X_train_pca, X_val_pca, X_test_pca

# Prepare tf Datasets
#

import tensorflow as tf

def prep_tf_datasets(X_train, X_val, y_train, y_val, batch_size):

    # Dimensions
    embedding_dimensions = X_train.shape[1]

    # Create tf datasets
    train_dataset_tf = tf.data.Dataset.from_tensor_slices((X_train, y_train.astype(np.float32)))
    val_dataset_tf = tf.data.Dataset.from_tensor_slices((X_val, y_val.astype(np.float32)))
    # test_dataset_tf = tf.data.Dataset.from_tensor_slices((X_test_pca, y_test.astype(np.float32)))

    # Shuffle, batch etc to improve gradient descent, efficiency ...
    buffer_size = max(1024, len(X_train) + len(X_val))

    train_dataset_tf = train_dataset_tf.shuffle(buffer_size=buffer_size)
    train_dataset_tf = train_dataset_tf.batch(batch_size)
    val_dataset_tf = val_dataset_tf.batch(batch_size)
    val_dataset_tf = val_dataset_tf.prefetch(1)

    print(train_dataset_tf)
    print(val_dataset_tf)

    return train_dataset_tf, val_dataset_tf
