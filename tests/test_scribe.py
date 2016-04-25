#!/usr/bin/env python3.5

import pytest
from scribe.LSA import LSA
from nltk.stem import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer

WORKERS = 8


class TestLSA(object):
    @pytest.fixture(scope='module')
    def docs(self):
        contents = 'fooffooo\nbarr bbbiiz bboooo\nbazzz fooffooo barr\nfooffooo barr'
        documents = LSA.read_raw_docs(contents.split('\n'), -1, 1)
        return documents

    @pytest.fixture(scope='module')
    def doclength(self, docs):
        return len(docs)

    @pytest.fixture(scope='module')
    def tokenizer(self):
        return TreebankWordTokenizer()

    @pytest.fixture(scope='module')
    def stemmer(self):
        return SnowballStemmer('english')

    def test_cleaner(self, tokenizer, stemmer):
        test_clean = lambda x: LSA.clean_text(x, tokenizer, stemmer)
        assert test_clean(['fooo']) == ['fooo']
        assert test_clean(['~!f*&%o)!)!oo{":?>']) == ['']
        assert test_clean(['32408732458 123981o304897345']) == ['32408732458 123981o304897345']
        assert test_clean(['foo bar']) == ['foo bar']
        assert test_clean(['f234872034ooo b3890ar0120']) == ['f234872034ooo b3890ar0120']
        assert test_clean(['!@#():foo b&$:baar*&@%']) == ['foo baar']

    def test_unique_words(self, docs):
        unique_words_result = LSA.unique_words(docs)
        words = ['barr', 'bazzz', 'bbbiiz', 'bboooo', 'fooffooo']
        assert sorted(unique_words_result.keys()) == words
