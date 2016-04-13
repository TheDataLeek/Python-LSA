#!/usr/bin/env python3.5

"""
Analyze embedding of documents in R^n space using paragraph vector algorithm

On the order of thousands of documents, not millions. Specifically implemented for
the associated Hillary dataset

Notes about implementation:
* Using google's skip gram model for embedding.
* Using hierarchical softmax
* Basically this will result in a matrix of word probabilities based on prior words
* NxN matrix, which given the size of our input data, is a little unrealistic
    * can't store the thing
    * can't visualize the thing
    * HIGH Dimensionality
* For ML stuffs, planning on visualizing each word in corpus in "word - word space"
    * "here are words in corpus represented in 'libya'-'benghazi' space,
        regressing on 'tiger'"
"""

from ..LSA import LSA
import random
import typing
import heapq
import numpy as np


def embedding(filename, workers):
    train_percent = 0.25
    train_data, test_data = get_data(filename, train_percent)
    words, word_counts, wordcount = ordered_words(train_data)
    encoding = Encoding(word_counts, wordcount)


def get_data(filename: str, train_percent: float):
    training = []
    testing  = []
    for doc in read_docs(filename):
        (testing if random.random() < train_percent else training).append(doc)
    return training, testing


def read_docs(filename: str) -> typing.Generator[str, str, str]:
    with open(filename, 'r') as ofile:
        for line in ofile.readlines():
            yield LSA.clean_words(line)


def ordered_words(docs: list):
    words = {}
    word_counts = {}
    index = 0
    total_words = 0
    for doc in docs:
        for word in doc:
            total_words += 1
            try:
                word_counts[word] += 1
            except KeyError:
                word_counts[word] = 1
                words[word] = index
                index += 1
    return words, word_counts, total_words


class Encoding(object):
    def __init__(self, word_counts, total_words):
        self.root = None
        self._queue = []
        self._words = word_counts
        self._total = total_words
        self._create()
        self._encoding = {}

    def _create(self):
        nodelist = []
        for word in self._words.keys():
            new_node = Node(self._words[word] / self._total)
            new_node.word = word
            heapq.heappush(self._queue, new_node)
            nodelist.append(new_node)
        while len(self._queue) > 1:
            node1 = heapq.heappop(self._queue)
            node2 = heapq.heappop(self._queue)
            new_node = Node(node1.freq + node2.freq)
            new_node.left = node1
            new_node.right = node2
            heapq.heappush(self._queue, new_node)
        self.root = heapq.heappop(self._queue)
        self._gen_codes(nodelist)

    def _gen_codes(self, nodelist):
        nodes = [self.root]
        while len(nodes) > 0:
            node = nodes.pop()
            if node.word is None:
                node.left.code = node.code + '0'
                node.right.code = node.code + '1'
                nodes.append(node.left)
                nodes.append(node.right)
        self._encoding = {n.word: n.code for n in nodelist}

    def _find_word(self, word):
        code = self._encoding[word]
        path = [self.root]
        for bit in code:
            if bit == '0':
                path.append(path[-1].left)
            else:
                path.append(path[-1].right)
        return path

    def softmax(self, word: str, input_word: str) -> float:
        sigma = lambda x : 1 / (1 + np.exp(-x))
        path = self._find_word(word)
        node = path[-1]
        vals = np.zeros(len(path) - 1, dtype=float)
        for i in range(len(path) - 1):  # every node but last
            vals[i] = sigma(0)
        return np.prod(vals)


class Node(object):
    __slots__ = ['word', 'freq', 'left', 'right', 'code']

    def __init__(self, freq):
        self.word = None
        self.freq = freq
        self.left = None
        self.right = None
        self.code = ''

    def __lt__(self, other):
        return self.freq < other.freq

    def __str__(self):
        return '{}:{}'.format(self.word, self.freq)
