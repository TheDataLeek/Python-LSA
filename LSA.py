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
import typing
import numpy as np
import scipy.io as scio
import scipy.sparse.linalg as ssl
from scipy.sparse import dok_matrix
from scipy.sparse import dok
from tqdm import tqdm
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

    docmatrix, documents = get_sparse_matrix(documents, words, args.workers)
    print('Calculated Sparse Matrix\nTime Elapsed: {}\n'.format(time.clock()))

    u, s, vt = ssl.svds(docmatrix.T, k=args.svdk)
    print('Calculated SVD Decomposition\nTime Elapsed: {}'.format(time.clock()))

    if args.save:
        output = {'u':u, 'd': np.diag(s), 'vt':vt,
                    'documents': np.array(documents, dtype=object),
                    'words': np.array(list(sorted(words.keys())), dtype=object)}
        print('Saving U: {}, S: {}, V.T: {}'.format(u.shape, s.shape, vt.shape))
        save_output(output)


@enforce.runtime_validation
def open_documents(filename: str, size: int) -> typing.Tuple[np.ndarray, int]:
    with open(filename, 'r') as datafile:
        lines = datafile.read().split('\n')
        documents = read_raw_docs(lines, size)
    doccount = len(documents)
    return documents, doccount


@enforce.runtime_validation
def read_raw_docs(lines: list, size: int) -> np.ndarray:
    if size == -1:
        size = len(lines)
    documents = np.empty(size, dtype=object)
    for i, line in enumerate(lines):
        if i >= size:
            break
        documents[i] = clean_text(line)
    return documents


@enforce.runtime_validation
def clean_text(line: str) -> str:
    return re.sub('[^a-z ]+', '', line.lower())


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
    words  = {}
    olddoc = None
    for doc in data:
        for word in doc.split(' '):
            if word != '':
                try:
                    words[word]['freq'] += 1
                    if doc != olddoc:
                        words[word]['doccount'] += 1
                except KeyError:
                    words[word] = {'freq': 1, 'doccount': 1}
        olddoc = doc
    return words


@enforce.runtime_validation
def get_sparse_matrix(documents: np.ndarray, words: dict, workers: int, weighting: typing.Optional[str]='default') -> typing.Tuple[dok.dok_matrix, np.ndarray]:
    """
    Parallelize Sparse Matrix Calculation

    :documents: list of document strings
    :words: dictionary of word frequencies
    :workers: number of workers

    :return: Sparse document term matrix
    """
    m         = len(documents)
    n         = len(words.keys())
    # Make sure we don't have more bins than workers
    workers   = m if m < workers else workers
    data_bins = np.array_split(documents, workers)
    docmatrix = dok_matrix((m, n), dtype=float)
    new_docs  = []
    offsets   = [len(data_bin) for data_bin in data_bins]
    coffset   = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(parse_docs, data_bins[i], words, len(documents)):i
                   for i in range(workers)}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           desc='Parsing Documents and Combining Arrays',
                           leave=True, total=workers):
            binnum = futures[future]
            # Because order is not preserved in threads, we need to make sure we add
            # the documents back in the correct order.
            for doc in data_bins[binnum]:
                new_docs.append(doc)
            # THIS IS THE BOTTLENECK
            for key, value in future.result().items():
                # TODO: Add more weight options
                weight = value if weighting == 'default' else 1
                docmatrix[key[0] + coffset, key[1]] = weight
            coffset += offsets[binnum]
    new_docs = np.array(new_docs, dtype=object)
    return docmatrix, new_docs


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
        for word in list(set(doc.split(' '))):
            if word != '':
                docmatrix[(i, wordref[word])] = weight(total_doc_count,
                                                       words[word]['doccount'],
                                                       words[word]['freq'])
    return docmatrix


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
