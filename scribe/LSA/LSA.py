#!/usr/bin/env python3

"""
Library for (level 2 optimized) Latent Sentiment Analysis
"""


import re
import math
import concurrent.futures
import logging
import typing
import numpy as np
import scipy.io as scio
from scipy.sparse import dok_matrix
from scipy.sparse import dok
import scipy.sparse.linalg as ssl
from tqdm import tqdm
import time
import enforce


def analyze(filename: str, workers: int, count: int, svdk: int, save: bool) -> None:
    """
    Manage analysis of document set
    """
    documents, doccount = open_documents(filename, count)
    print('Program Start. Loaded Data. Time Elapsed: {}\n'.format(time.clock()))
    logging.info('Loaded Data. Time Elapsed: {}'.format(time.clock()))

    words = get_unique_words(documents, workers)
    wordcount = len(words.keys())
    topwords = ','.join([w for w, s in sorted(words.items(),
                                              key=lambda tup: -tup[1]['freq'])[:20]])

    logging.info('Found Word Frequencies')
    logging.info('{} Documents (m) by {} Unique Words (n)'.format(doccount, wordcount))
    logging.info('Top 20 Most Frequent Words:{}'.format(topwords))
    logging.info('Time Elapsed: {}'.format(time.clock()))

    print(('Found Word Frequencies\n'
           '\n{} Documents (m) by {} Unique Words (n)\n\n'
           'Top 20 Most Frequent Words:{}\n'
           'Time Elapsed: {}\n').format(doccount,
                                        wordcount,
                                        topwords,
                                        time.clock()))

    docmatrix, documents = get_sparse_matrix(documents, words, workers)
    print('Calculated Sparse Matrix\nTime Elapsed: {}\n'.format(time.clock()))
    logging.info('Calculated Sparse Matrix. Time Elapsed: {}'.format(time.clock()))

    u, s, vt, wordlist = matrix_comparison(docmatrix, svdk, words, documents)
    print('Calculated SVD Decomposition\nTime Elapsed: {}'.format(time.clock()))
    logging.info('Calculated SVD Decomposition. Time Elapsed: {}'.format(time.clock()))

    if save:
        output = {'u':u, 'd': np.diag(s), 'vt':vt,
                  'documents': np.array(documents, dtype=object),
                  'words': wordlist}
        print('Saving U: {}, S: {}, V.T: {}'.format(u.shape, s.shape, vt.shape))
        save_output(output)


def matrix_comparison(docmatrix, k, words, documents):
    u, s, vt = ssl.svds(docmatrix.T, k=k)
    wordlist = np.array(list(sorted(words.keys())), dtype=object)
    d = np.diag(s)

    num_docs = vt.shape[1]
    num_words = u.shape[0]

    word_mat = u @ d    # python 3.5 syntax for dot product
    doc_mat = d @ vt

    query = input('Enter the query word: ').lower()   # we can chain the lower()

    index = -1
    for i, word in enumerate(wordlist):  # Loog has runtime O(n)
        if word == query:
            index = i
            break   # Breaks us out of loop so we don't need to iterate

    if index == -1:   # If we didn't find word in the corpus, don't analyze
        print('Invalid Word -- Not in Corpus')
    else:
        q = word_mat[index]

        rank = np.zeros(num_docs)  # pre-initialize array for speed
        for i in range(num_docs):
            rank[i] = (doc_mat[:, i] @ q) / (np.linalg.norm(doc_mat[:, i]) * np.linalg.norm(q))

        r = sorted(range(len(rank)), key=lambda x: rank[x])
        print('\n')
        print(documents[r[-1]])
        print(documents[r[-2]])

    return u, s, vt, wordlist


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
        documents[i] = str(clean_text(line))
    return documents


def clean_text(line: str) -> str:
    return re.sub('[^a-z ]+', '', line.lower())


def clean_words(line: str) -> typing.Generator:
    for word in clean_text(line).split(' '):
        if word != '':
            yield word


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


def weight(total_doc_count: int, doccount: int, wordfreq: int) -> float:
    """
    Weighting function for Document Term Matrix.

    tf-idf => https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    """
    return math.log(total_doc_count / doccount) * wordfreq


@enforce.runtime_validation
def get_sparse_matrix(documents: np.ndarray, words: dict, workers: int, weighting: typing.Any=weight) -> typing.Tuple[dok.dok_matrix, np.ndarray]:
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
    new_docs  = np.empty(len(documents), dtype=object)
    offsets   = [len(data_bin) for data_bin in data_bins]
    coffset   = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(parse_docs, data_bins[i], words, len(documents), weight):i
                   for i in range(workers)}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           desc='Parsing Documents and Combining Arrays',
                           leave=True, total=workers):
            binnum = futures[future]
            # Because order is not preserved in threads, we need to make sure we add
            # the documents back in the correct order.
            for i, doc in enumerate(data_bins[binnum]):
                new_docs[coffset + i] = doc
            # THIS IS THE BOTTLENECK
            for key, value in future.result().items():
                docmatrix[key[0] + coffset, key[1]] = value
            coffset += offsets[binnum]
    return docmatrix, new_docs


@enforce.runtime_validation
def parse_docs(data: np.ndarray, words: dict, doc_count: int, weight_func: typing.Any) -> dict:
    """
    Parallelize Sparse Matrix Calculation

    :data: list of document strings
    :words: dictionary of word frequencies
    :total_doc_count: total number of documents (for tf-idf)
    :weight_func: weighting function for code

    :return: Basically sparse array with weighted values
    """
    m = len(data)
    n = len(words.keys())
    docmatrix = {}
    wordref = {w:i for i, w in enumerate(sorted(words.keys()))}
    for i, doc in enumerate(data):
        for word in list(set(doc.split(' '))):
            if word != '':
                docmatrix[(i, wordref[word])] = weight_func(doc_count,
                                                            words[word]['doccount'],
                                                            words[word]['freq'])
    return docmatrix


@enforce.runtime_validation
def save_output(output: dict) -> None:
    scio.savemat('output.mat', output)

