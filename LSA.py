#!/usr/bin/env python3

import sys
import re
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.sparse as scs
import scipy.sparse.linalg as ssl
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix

from tqdm import tqdm

WORKERS = 8

def main():
    data = pd.read_csv('./data/JEOPARDY_CSV.csv')
    documents = data[[' Question']].values[:, 0]

    doccount = len(data)

    words = get_unique_words(documents)
    wordcount = len(words.keys())

    print('{} Documents (m) by {} Unique Words (n)\n\nTop 100 Most Frequent Words:{}'.format(
            doccount, wordcount, ','.join([tup[0] for tup in sorted(words.items(), key=lambda tup: -tup[1])[:100]])))

    return

    docmatrix = dok_matrix((m, n), dtype=float)   # m-docs, n-unique words

    ndocterm, wordref = populate_doc_matrix(docmatrix, words, wordfreq,
                                    data[[' Question', ' Answer']].values)

    ndocterm

    u, s, vt = ssl.svds(ndocterm.T, k=20)


def get_unique_words(documents):
    data_bins = np.array_split(documents, 8)   # TODO: adjustable bins
    wordlist = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(unique_words, data_bins[i]):i for i in range(WORKERS)}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           desc='Determining Unique Words', leave=True, total=WORKERS):
            i = futures[future]
            for word, freq in future.result().items():
                try:
                    wordlist[word] += freq
                except KeyError:
                    wordlist[word] = freq
    return wordlist


def unique_words(data):
    words = {}
    for doc in data:
        for word in doc.split(' '):
            cword = re.sub('[^a-z]+', '', word.lower())
            if cword != '':
                try:
                    words[cword] += 1
                except KeyError:
                    words[cword] = 1
    return words

# Use tf-idf
# https://en.wikipedia.org/wiki/Tf%E2%80%93idf
def populate_doc_matrix(docmatrix, wordlist, word_freq, data):
    n = len(data)   # number of documents
    # construct word index first
    # This tells us (for any word) what index it is in in document
    print('Constructing Word Reference')
    wordref = {}
    for i in range(len(wordlist)):
        wordref[wordlist[i]] = i
    # Now populate sparse matrix
    print('Populating Sparse Matrix')
    for i in range(n):
        for j in range(2):
            words = [w.lower() for w in data[i, j].split(' ') if w != '']
            m = len(words)
            for k in range(m):
                word = words[k]
                cword = ''
                for char in word:
                    if char in alphabet:
                        cword += char
                if cword != '':
                    docmatrix[i, wordref[cword]] += 1
    # finish weighting
    print('Weighting Matrix')
    m, n = docmatrix.shape
    weighted_docmatrix = dok_matrix((m, n), dtype=float)
    for i in range(n):
        weighted_docmatrix[:, i] = docmatrix[:, i] * np.log(m / word_freq[wordlist[i]])
    return weighted_docmatrix, wordref

if __name__ == '__main__':
    sys.exit(main())
