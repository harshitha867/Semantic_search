from sklearn.metrics.pairwise import cosine_similarity
from embeddings import model
import numpy as np


def semantic_search(query, chunks, embeddings, top_k=3):

    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append(
            (
                chunks[idx],
                similarities[idx]
            )
        )

    return results