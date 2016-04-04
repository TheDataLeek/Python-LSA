#!/usr/bin/env python3


"""
Perform LSA on given set of text.

See README.md for details
"""


import sys
import re
import math
import time
import argparse
import concurrent.futures
import numpy as np
import scipy.io as scio
import scipy.sparse.linalg as ssl
from scipy.sparse import dok_matrix
from scipy.sparse import dok
from tqdm import tqdm
import numba
import enforce


def main():
    """ Manage Execution """
    args = get_args()

    documents, doccount = open_documents(args.filename, args.count)
    print('Program Start. Loaded Data. Time Elapsed: {}\n'.format(time.clock()))

    words = get_unique_words(documents, args.workers)
    wordcount = len(words.keys())
    topwords = ','.join([w for w, s in sorted(words.items(),
                                              key=lambda tup: -tup[1]['freq'])[:100]])

    print(('Found Word Frequencies\n'
           '\n{} Documents (m) by {} Unique Words (n)\n\n'
           'Top 100 Most Frequent Words:{}\n'
           'Time Elapsed: {}\n').format(doccount,
                                        wordcount,
                                        topwords,
                                        time.clock()))

    docmatrix = get_sparse_matrix(documents, words, args.workers)
    print('Calculated Sparse Matrix\nTime Elapsed: {}\n'.format(time.clock()))

    u, s, vt = ssl.svds(docmatrix.T, k=args.svdk)
    print('Calculated SVD Decomposition\nTime Elapsed: {}'.format(time.clock()))

    if args.save:
        output = {'u':u, 'd': np.diag(s), 'vt':vt, 'words': list(words.keys())}
        save_output(output)


@enforce.runtime_validation
def open_documents(filename: str, size: int) -> tuple:
    with open(filename, 'r') as datafile:
        lines = datafile.read().split('\n')
        if size == -1:
            size = len(lines)
        documents = np.empty(size, dtype=object)
        for i, line in enumerate(lines):
            if i >= size:
                break
            documents[i] = line
    doccount = len(documents)
    return documents, doccount


@enforce.runtime_validation
def get_unique_words(documents: np.ndarray, workers: int) -> dict:
    """
    Parallelize Unique Word Calculation

    :documents: list of document strings
    :workers: number of workers

    :return: dictionary of word frequencies
    """
    data_bins = np.array_split(documents, workers)
    wordlist = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(unique_words, data_bins[i]):i for i in range(workers)}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           desc='Determining Unique Words', leave=True, total=workers):
            for word, stats in future.result().items():
                try:
                    wordlist[word]['freq'] += stats['freq']
                    wordlist[word]['doccount'] += stats['doccount']
                except KeyError:
                    wordlist[word] = {'freq':stats['freq'], 'doccount':stats['doccount']}
    return wordlist


@enforce.runtime_validation
def unique_words(data: np.ndarray) -> dict:
    """
    Finds unique word frequencies in documents

    :data: list of document strings

    :return: dictionary of word frequencies
    """
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


@enforce.runtime_validation
def get_sparse_matrix(documents: np.ndarray, words: dict, workers: int) -> dok.dok_matrix:
    """
    Parallelize Sparse Matrix Calculation

    :documents: list of document strings
    :words: dictionary of word frequencies
    :workers: number of workers

    :return: Sparse document term matrix
    """
    m = len(documents)
    n = len(words.keys())
    data_bins = np.array_split(documents, workers)
    docmatrix = dok_matrix((m, n), dtype=float)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(parse_docs, data_bins[i], words, len(documents)):i
                   for i in range(workers)}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           desc='Parsing Documents and Combining Arrays',
                           leave=True, total=workers):
            # THIS IS THE BOTTLENECK
            for key, value in future.result().items():
                docmatrix[key[0], key[1]] = value
    return docmatrix


@enforce.runtime_validation
def parse_docs(data: np.ndarray, words: dict, total_doc_count: int) -> dict:
    """
    Parallelize Sparse Matrix Calculation

    :data: list of document strings
    :words: dictionary of word frequencies
    :total_doc_count: total number of documents (for tf-idf)

    :return: Basically sparse array with weighted values
    """
    m = len(data)
    n = len(words.keys())
    docmatrix = {}
    wordref = {w:i for i, w in enumerate(sorted(words.keys()))}
    for i, doc in enumerate(data):
        for word in list(set([re.sub('[^a-z]+', '', w.lower()) for w in doc.split(' ')])):
            if word != '':
                docmatrix[(i, wordref[word])] = weight(total_doc_count,
                                                       words[word]['doccount'],
                                                       words[word]['freq'])
    return docmatrix


@numba.jit
def weight(total_doc_count: int, doccount: int, wordfreq: int) -> float:
    """
    Weighting function for Document Term Matrix.

    tf-idf => https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    """
    return math.log(total_doc_count / doccount) * wordfreq


@enforce.runtime_validation
def save_output(output: dict) -> None:
    scio.savemat('output.mat', output)


@enforce.runtime_validation
def get_args() -> argparse.Namespace:
    """
    Get Command line Arguments

    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workers', type=int, default=32,
                        help=('Number of workers to use for multiprocessing'))
    parser.add_argument('-c', '--count', type=int, default=-1,
                        help=('Number of documents to use from original set'))
    parser.add_argument('-k', '--svdk', type=int, default=20,
                        help=('SVD Degree'))
    parser.add_argument('-f', '--filename', type=str, default='./data/jeopardy.csv',
                        help=('File to use for analysis'))
    parser.add_argument('-s', '--save', action='store_true', default=False,
                        help=('Save output in .mat file.'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sys.exit(main())
