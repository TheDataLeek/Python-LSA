#!/usr/bin/env python3.5

import LSA
import numpy as np
import pytest

class TestLSA(object):
    def test_unique(self):
        words = sorted(['foo', 'bar', 'biz', 'baz'])
        docs = np.random.choice(words, size=1000)
        unique_words = LSA.unique_words(docs)
        assert sorted(unique_words.keys()) == words
