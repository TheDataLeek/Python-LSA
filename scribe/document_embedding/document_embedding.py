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


def embedding(filename, workers):
    savefile = '../data/embedding.npy'
    if not os.path.isfile(savefile):
        documents, doccount = LSA.open_documents(filename, -1)
        words = LSA.get_unique_words(documents, workers)
        docmatrix, documents = LSA.get_sparse_matrix(documents, words, workers)

        u, s, vt = ssl.svds(docmatrix, k=2)

        np.save(savefile, u)
    else:
        u = np.load(savefile)

    print(u.shape)

    plt.figure()
    plt.scatter(u[:, 0], u[:, 1])
    plt.show()
