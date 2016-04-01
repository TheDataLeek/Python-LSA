#!/usr/bin/env python3

import sys
import re
import math
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.sparse as scs
import scipy.sparse.linalg as ssl
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix

from tqdm import tqdm

WORKERS = 32

def main():
    data = pd.read_csv('./data/JEOPARDY_CSV.csv')
    #documents = data[[' Question']].values[:, 0]
    documents = data[[' Question']].values[:10000, 0]

    doccount = len(documents)

    words = get_unique_words(documents)
    wordcount = len(words.keys())

    print('\n{} Documents (m) by {} Unique Words (n)\n\nTop 100 Most Frequent Words:{}\n'.format(
            doccount, wordcount, ','.join([w for w, s in sorted(words.items(), key=lambda tup: -tup[1]['freq'])[:100]])))

    docmatrix = get_sparse_matrix(documents, words)

    print('Calculating SVD Decomposition')
    u, s, vt = ssl.svds(docmatrix, k=20)


def get_sparse_matrix(documents, words):
    m = len(documents)
    n = len(words.keys())
    data_bins = np.array_split(documents, WORKERS)   # TODO: adjustable bins
    docmatrix = dok_matrix((m, n), dtype=float)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(parse_docs, data_bins[i], words, len(documents)):i for i in range(WORKERS)}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           desc='Parsing Documents and Combining Arrays', leave=True, total=WORKERS):
            # THIS IS THE BOTTLENECK
            for key, value in future.result().items():
                docmatrix[key[0], key[1]] = value
    return docmatrix


def parse_docs(data, words, total_doc_count):
    m = len(data)
    n = len(words.keys())
    docmatrix = {}
    wordref = {w:i for i, w in enumerate(sorted(words.keys()))}
    for i, doc in enumerate(data):
        for word in list(set([re.sub('[^a-z]+', '', w.lower()) for w in doc.split(' ')])):
            if word != '':
                # tf-idf https://en.wikipedia.org/wiki/Tf%E2%80%93idf
                docmatrix[(i, wordref[word])] = math.log(total_doc_count / (words[word]['doccount'])) * words[word]['freq']
    return docmatrix


def get_unique_words(documents):
    data_bins = np.array_split(documents, WORKERS)   # TODO: adjustable bins
    wordlist = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(unique_words, data_bins[i]):i for i in range(WORKERS)}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           desc='Determining Unique Words', leave=True, total=WORKERS):
            for word, stats in future.result().items():
                try:
                    wordlist[word]['freq'] += stats['freq']
                    wordlist[word]['doccount'] += stats['doccount']
                except KeyError:
                    wordlist[word] = {'freq':stats['freq'], 'doccount':stats['doccount']}
    return wordlist


def unique_words(data):
    words = {}
    olddoc = None
    for doc in data:
        for word in doc.split(' '):
            cword = re.sub('[^a-z]+', '', word.lower())
            if cword != '':
                try:
                    words[cword]['freq'] += 1
                    if doc != olddoc:
                        words[cword]['doccount'] += 1
                except KeyError:
                    words[cword] = {'freq':1, 'doccount':1}
        olddoc = doc
    return words


if __name__ == '__main__':
    sys.exit(main())
