#!/usr/bin/env python3.5

import LSA
import numpy as np
import pytest

WORKERS = 8

class TestLSA(object):
    @pytest.fixture(scope='module')
    def docs(self):
        contents = 'foo\nbar biz boo\nbaz foo bar\nfoo bar'
        documents = LSA.read_raw_docs(contents.split('\n'), -1)
        return documents

    @pytest.fixture(scope='module')
    def doclength(self, docs):
        return len(docs)

    def test_unique(self):
        words = sorted(['foo', 'bar', 'biz', 'baz'])
        docs = np.random.choice(words, size=1000)
        unique_words = LSA.unique_words(docs)
        assert sorted(unique_words.keys()) == words

    def test_sparse_matrix(self, docs, doclength):
        """
        Sparse matrix should be of form (transposed)

            1 2 3 4
        bar 0 1 1 1
        baz 0 0 1 0
        biz 0 1 0 0
        boo 0 1 0 0
        foo 1 0 1 1
        """
        words = LSA.get_unique_words(docs, WORKERS)
        wordcount = len(words.keys())
        alphawords = sorted(list(words.keys()))
        docmatrix, newdocs = LSA.get_sparse_matrix(docs, words,
                                                   WORKERS, weighting='freq')

        # Transpose docmatrix
        docmatrix = docmatrix.T

        # first just make sure it's the right shape
        assert docmatrix.shape == (wordcount, doclength)

        # Now make sure it's the right shape
        assert (docmatrix.todense() == np.array([[0, 1, 1, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 1, 0, 0],
                                                 [0, 1, 0, 0],
                                                 [1, 0, 1, 1]])).all()

    def test_clean_text_word(self):
        word = '!@#*(&F)(O&!{}:"><ObAR1230875'
        assert LSA.clean_text(word) == 'foobar'

    def test_clean_text_sentence(self):
        sentence = '!@#*(&F) (O&!{}:"><O bAR123 testfoo0875'
        assert LSA.clean_text(sentence) == 'f oo bar testfoo'

