#!/usr/bin/env python3.5

"""
Analyze embedding of documents in R^n space using paragraph vector algorithm
"""

from ..LSA import LSA


def embedding():
    filename = '../data/hillary/cleanhillary.txt'
    docs = LSA.open_documents(filename, 10)
    print(docs)

