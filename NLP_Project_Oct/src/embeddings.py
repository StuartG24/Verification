# Embedding Functions
#

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

def create_embeddings(sentences):
    # TODO: Understand why vector alignment is needed and how it is performed, see Antonio
    # See https://www.kaggle.com/discussions/general/566301

    # SBERT with all-MiniLM-L6-V2, a small and fast model, 385 dimensions
    # ANTONIO: encoding_models = {'all-MiniLM-L6-v2': 'sbert22M'}
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sbert_model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)

    # Allign vectors to the axis and rotate
    u, s, vh = np.linalg.svd(a=embeddings)
    align_mat = np.linalg.solve(a=vh, b=np.eye(len(embeddings[0])))
    embeddings = np.matmul(embeddings, align_mat)
    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings
