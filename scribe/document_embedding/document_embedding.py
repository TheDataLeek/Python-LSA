#!/usr/bin/env python3.5

"""
Analyze embedding of documents in R^n space using paragraph vector algorithm
"""

from ..LSA import LSA
import gensim
from gensim import models
from gensim.models import doc2vec
from gensim.models import Doc2Vec


def embedding(filename, workers):
    """
    Use Gensim to do document embedding.
    """
    modelname = filename + '.model'
    documents = doc2vec.TaggedLineDocument(filename)
    model = gensim.models.Doc2Vec(documents, workers=workers)

    print(model.most_similar())
