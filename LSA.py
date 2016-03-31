#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.sparse as scs
import scipy.sparse.linalg as ssl
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix

from tqdm import tqdm


alphabet = 'abcdefghijklmnopqrstuvwxyz'


def main():
    data = pd.read_csv('./data/JEOPARDY_CSV.csv')
    data = data[:1000]

    m = len(data)

    words, n, wordfreq = unique_words(list(np.concatenate((data[[' Question']].values[:, 0],
                                      data[[' Answer']].values[:, 0]))))

    print('{} Documents (m) by {} Unique Words (n)\n\nTop 100 Most Frequent Words:{}'.format(
            m, n, ','.join([tup[0] for tup in sorted(wordfreq.items(), key=lambda tup: -tup[1])[:100]])))


    docmatrix = dok_matrix((m, n), dtype=float)   # m-docs, n-unique words

    ndocterm, wordref = populate_doc_matrix(docmatrix, words, wordfreq,
                                    data[[' Question', ' Answer']].values)

    ndocterm

    u, s, vt = ssl.svds(ndocterm.T, k=20)


def unique_words(sentences):
    words = {}
    n = len(sentences)
    for i in range(n):
        sent_list = [w.lower() for w in sentences[i].split(' ')]
        clean_sent_list = []
        for j in range(len(sent_list)):
            newword = ''
            for char in sent_list[j]:
                if char in alphabet:
                    newword += char
            clean_sent_list.append(newword)
        for word in clean_sent_list:
            if word != '':
                try:
                    words[word] += 1
                except KeyError:
                    words[word] = 1
    wordlist = sorted(words.keys())
    return wordlist, len(wordlist), words

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
