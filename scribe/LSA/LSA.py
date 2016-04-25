#!/usr/bin/env python3

"""
Library for (level 2 optimized) Latent Sentiment Analysis
"""


import re
import math
import concurrent.futures
import typing
import numpy as np
from scipy import spatial
import scipy.stats as sct
from scipy.sparse import dok_matrix
from scipy.sparse import dok
import scipy.sparse.linalg as ssl
from tqdm import tqdm
import time
import enforce
from nltk.stem import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer


np.set_printoptions(linewidth=160)


def analyze(filename: str, workers: int, count: int, svdk: int, save: bool, output: bool) -> None:
    """
    Manage analysis of document set
    """
    documents, doccount = open_documents(filename, count, workers)
    print('Program Start. Loaded Data. Time Elapsed: {}\n'.format(time.clock()))

    if len(documents) > 100:
        output = False

    if output:
        print(documents)

    words = get_unique_words(documents, workers)
    wordcount = len(words.keys())
    topwords = ','.join([w for w, s in sorted(words.items(),
                                              key=lambda tup: -tup[1]['freq'])[:20]])

    print(('Found Word Frequencies\n'
           '\n{} Documents (m) by {} Unique Words (n)\n\n'
           'Top 20 Most Frequent Words:{}\n'
           'Time Elapsed: {}\n').format(doccount,
                                        wordcount,
                                        topwords,
                                        time.clock()))

    if output:
        for word, freqs in words.items():
            print('{} => {}'.format(word, freqs))

    docmatrix, documents = get_sparse_matrix(documents, words, workers)
    print('Calculated Sparse Matrix\nTime Elapsed: {}\n'.format(time.clock()))

    if output:
        docs = docmatrix.T
        comparisons = np.zeros((len(documents), len(documents)))
        for i in range(len(documents)):
            for j in range(len(documents)):
                comparisons[i, j] = 1 - spatial.distance.cosine(docs[:, i].todense(), docs[:, j].todense())
        print(comparisons)

    u, s, vt = decomposition(docmatrix, svdk)
    print('Calculated SVD Decomposition\nTime Elapsed: {}'.format(time.clock()))

    if output:
        docs = np.diag(s) @ vt
        comparisons = np.zeros((len(documents), len(documents)))
        for i in range(len(documents)):
            for j in range(len(documents)):
                comparisons[i, j] = 1 - spatial.distance.cosine(docs[:, i], docs[:, j])
        print(comparisons)

    while True:
        try:
            selection = input('(w)ords or (d)ocuments? ').lower()
            if selection == 'w':
                matrix_comparison(u, s, vt, words, documents, output)
            elif selection == 'd':
                doc_comparisons(u, s, vt, documents, output)
            elif selection == 'exit':
                break
        except (KeyboardInterrupt, EOFError):
            break


def decomposition(docmatrix, k):
    u, s, vt = ssl.svds(docmatrix.T, k=k)
    return u, s, vt


def matrix_comparison(u, s, vt, words, documents, output):
    wordlist = np.array(list(sorted(words.keys())), dtype=object)
    word_indices = {w:i for i, w in enumerate(wordlist)}
    d = np.diag(s)

    num_docs = vt.shape[1]
    num_words = u.shape[0]

    word_mat = u @ d    # python 3.5 syntax for dot product
    doc_mat = d @ vt

    tokenizer = TreebankWordTokenizer()
    stemmer = SnowballStemmer('english')
    query = [w for w in
             clean_text([input('Enter the query: ')], tokenizer, stemmer)[0].split(' ')
             if w != '']

    indices = []
    error = False
    for word in query:
        try:
            indices.append(word_indices[word])
        except KeyError:
            print('Invalid Word: {} -- Not in Corpus'.format(word))
            error = True
            break

    if not error:
        q = np.mean([word_mat[index] for index in indices], axis=0)

        rank = np.zeros(num_docs)  # pre-initialize array for speed
        for i in range(num_docs):
            rank[i] = 1 - spatial.distance.cosine(doc_mat[:, i], q)

        r = sorted(range(len(rank)), key=lambda x: rank[x])

        if output:
            for i in range(-1, -len(r) - 1, -1):
                print('{})\t{}'.format(np.abs(i), documents[r[i]]))
        else:
            for i in range(-1, -10, -1):
                print('{})\t{}'.format(np.abs(i), documents[r[i]]))


def doc_comparisons(u, s, vt, documents, output):
    d = np.diag(s)
    num_docs = vt.shape[1]
    doc_mat = d @ vt

    error = False

    try:
        query_str = 'Enter the document number you wish to query (between 0 and {} inclusive): '
        index = int(input(query_str.format(len(documents) - 1)))
        if index >= len(documents) or index < 0:
            error = True
        print('You Queried: {}'.format(documents[index]))
    except ValueError:
        print("Insert Valid Number")
        error = True

    try:
        method = int(input("Enter 1 for Spearmans and 2 for Cosine Similarity: "))
        if method not in [1, 2]:
            raise ValueError
    except ValueError:
        error = True

    if not error:
        q = doc_mat[:, index]

        print(q)

        rank = np.zeros(num_docs)
        if method == 1:
            distance_func = lambda a, b: sct.spearmanr(a, b)[0]
        elif method == 2:
            distance_func = lambda a, b: 1 - spatial.distance.cosine(a, b)

        for i in range(num_docs):
            rank[i] = distance_func(doc_mat[:, i], q)

        r = sorted(range(len(rank)), key=lambda x: rank[x])

        if output:
            for i in range(-1, -len(r) - 1, -1):
                print('{})\t{}'.format(np.abs(i), documents[r[i]]))
        else:
            for i in range(-1, -10, -1):
                print('{})\t{}'.format(np.abs(i), documents[r[i]]))


@enforce.runtime_validation
def open_documents(filename: str, size: int, workers: int) -> typing.Tuple[np.ndarray, int]:
    with open(filename, 'r') as datafile:
        lines = datafile.read().split('\n')[:-1]
        documents = read_raw_docs(lines, size, workers)
    doccount = len(documents)
    return documents, doccount


@enforce.runtime_validation
def read_raw_docs(lines: list, size: int, workers: int) -> np.ndarray:
    if size == -1:
        size = len(lines)
    lines = lines[:size]
    documents = np.empty(size, dtype=object)
    tokenizer = TreebankWordTokenizer()
    stemmer = SnowballStemmer('english')
    linebins = np.array_split(lines, workers)
    offset = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(clean_text, linebins[i], tokenizer, stemmer):i
                   for i in range(workers)}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           desc='Tokenizing Documents', total=workers, leave=True):
            index = futures[future]
            for i, clean_line in enumerate(future.result()):
                documents[offset + i] = clean_line
            offset += len(future.result())
    return documents


def clean_text(lines: np.ndarray,
               tokenizer: TreebankWordTokenizer,
               stemmer: SnowballStemmer) -> np.ndarray:
    """
    This module is responsible for converting a document into a cleaned version using nltk
    1. Tokenize using TreeBankTokenizer: http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.treebank
    2. Remove punctuation
    3. Remove words that are 1-3 characters long
    4. Stem using Snowball stemmer: http://www.nltk.org/howto/stem.html
    5. Join back together
    """
    clean_lines = np.empty(len(lines), dtype=object)
    for i, line in enumerate(lines):
        tokens = tokenizer.tokenize(line.lower())
        #stemmed_tokens = [stemmer.stem(w) for w in
        #                  [re.sub('[^a-z0-9 ]+', '', word) for word in tokens]
        #                  if len(w) > 2]
        stemmed_tokens = [w for w in
                          [re.sub('[^a-z0-9 ]+', '', word) for word in tokens]
                          if len(w) > 2]
        clean_lines[i] = ' '.join(stemmed_tokens)
    return clean_lines


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
    for doc in data:
        docwords_list = [w for w in doc.split(' ') if w != '']
        docwords_set = set(docwords_list)
        for word in docwords_list:
            try:
                words[word]['freq'] += 1
            except KeyError:
                words[word] = {'freq': 1, 'doccount': 0}
        for word in docwords_set:
            words[word]['doccount'] += 1
    return words


def weight(total_doc_count: int, doccount: int, wordfreq: int) -> float:
    """
    Weighting function for Document Term Matrix.

    tf-idf => https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    """
    return (1 + np.log(wordfreq)) * (np.log(total_doc_count / doccount))


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
        futures = {executor.submit(parse_docs,
                                   data_bins[i],
                                   words,
                                   len(documents),
                                   weight):i
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

