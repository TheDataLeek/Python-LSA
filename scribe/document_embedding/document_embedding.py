#!/usr/bin/env python3.5

"""
Analyze embedding of documents in R^n space using paragraph vector algorithm

On the order of thousands of documents, not millions. Specifically implemented for
the associated Hillary dataset
"""

import os
from ..LSA import LSA
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as ssl
from gensim import models
import typing


def embedding(filename, workers):
    savefile = '../data/embedding.npy'
    if not os.path.isfile(savefile):
        documents, doccount = LSA.open_documents(filename, -1)
        words = LSA.get_unique_words(documents, workers)
        docmatrix, documents = LSA.get_sparse_matrix(documents, words, workers)

        u, s, vt = ssl.svds(docmatrix, k=2)

        np.save(savefile, np.array([u, s, vt]))
    else:
        decomposition = np.load(savefile)
        u = decomposition[0]
        s = decomposition[1]
        vt = decomposition[2]

    mincount = 3
    model = models.Word2Vec(models.word2vec.LineSentence(filename),
                            sg=1, size=2, min_count=mincount, workers=8)
    documents, doccount = LSA.open_documents(filename, -1)
    words = sorted([w for w, freq in
                    LSA.get_unique_words(documents, workers).items()
                    if freq['freq'] > mincount])
    model.init_sims(replace=True)
    data = np.zeros((len(words), 2))
    for i, w in enumerate(words):
        data[i] = model[w]

    data = k_means(data, k=10)

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=100, c=data[:, -1], alpha=0.5)
    plt.show()


def k_means(data: np.ndarray, k: typing.Optional[int]=3,
            iterations: typing.Optional[int]=1000) -> np.ndarray:
    """
    K-Means clustering algorithm

    Using initial method of random sets - Forgy Method or Random Partition
    """
    mean_choices = np.random.choice(np.arange(len(data)), size=k, replace=False)
    means = data[mean_choices]
    data_means = np.zeros((data.shape[0], data.shape[1] + 1))
    data_means[:, :-1] = data
    vals = np.zeros((len(data), k))
    for i in range(iterations):
        for j in range(k):
            vals[:, j] = np.sum((data - means[j])**2, axis=1)
        data_means[:, -1] = vals.argmin(axis=1)
        for j in range(k):
            means[j] = data_means[data_means[:, -1] == j][:, :-1].mean(axis=0)
    return data_means



