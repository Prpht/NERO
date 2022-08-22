import numpy as np


def add_embeddings(embedding_a: np.array, embedding_b: np.array) -> np.array:
    """Calculates the sum of two embeddings, creating a larger array if necessary."""
    order_a, degree_a = embedding_a.shape
    order_b, degree_b = embedding_b.shape
    order, degree = max(order_a, order_b), max(degree_a, degree_b)
    if order_a == order and degree_a == degree:
        embedding_a[:embedding_b.shape[0], :embedding_b.shape[1]] += embedding_b
        return embedding_a
    elif order_b == order and degree_b == degree:
        embedding_b[:embedding_a.shape[0], :embedding_a.shape[1]] += embedding_a
        return embedding_b
    else:
        new_embedding = np.zeros((max(order_a, order_b), max(degree_a, degree_b)))
        new_embedding[:embedding_a.shape[0], :embedding_a.shape[1]] += embedding_a
        new_embedding[:embedding_b.shape[0], :embedding_b.shape[1]] += embedding_b
        return new_embedding


def add_embeddings_3d(embedding_a: np.array, embedding_b: np.array) -> np.array:
    """Calculates the sum of two 3D embeddings, creating a larger array if necessary."""
    order_a = embedding_a.shape[0]
    order_b = embedding_b.shape[0]
    order = max(order_a, order_b)
    if order_a == order:
        embedding_a[:embedding_b.shape[0], :, :] += embedding_b
        return embedding_a
    elif order_b == order:
        embedding_b[:embedding_a.shape[0], :, :] += embedding_a
        return embedding_b


def add_embeddings_4d(embedding_a: np.array, embedding_b: np.array) -> np.array:
    """Calculates the sum of two 4D embeddings, creating a larger array if necessary."""
    order_a = embedding_a.shape[0]
    order_b = embedding_b.shape[0]
    order = max(order_a, order_b)
    if order_a == order:
        embedding_a[:embedding_b.shape[0], :, :, :] += embedding_b
        return embedding_a
    elif order_b == order:
        embedding_b[:embedding_a.shape[0], :, :, :] += embedding_a
        return embedding_b
