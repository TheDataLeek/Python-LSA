#!/usr/bin/env python3.5

import numpy as np
import pytest

from scribe.LSA.LSA import clean_text
from scribe.LSA.LSA import unique_words
from scribe.LSA.LSA import read_raw_docs
from scribe.LSA.LSA import parse_docs

WORKERS = 8


class TestLSA(object):
    @pytest.fixture(scope='module')
    def docs(self):
        contents = 'foo\nbar biz boo\nbaz foo bar\nfoo bar'
        documents = read_raw_docs(contents.split('\n'), -1)
        return documents

    @pytest.fixture(scope='module')
    def doclength(self, docs):
        return len(docs)

    def test_cleaner(self):
        assert clean_text('foo') == 'foo'
        assert clean_text('~!f*&%o)!)!o{":?>') == 'foo'
        assert clean_text('3240873245872345087foo123981304897345') == 'foo'
        assert clean_text('foo bar') == 'foo bar'
        assert clean_text('f234872034oo b3890ar0120') == 'foo bar'
        assert clean_text('!@#():foo b&$:ar*&@%') == 'foo bar'

    def test_clean_text_word(self):
        word = '!@#*(&F)(O&!{}:"><ObAR1230875'
        assert clean_text(word) == 'foobar'

    def test_clean_text_sentence(self):
        sentence = '!@#*(&F) (O&!{}:"><O bAR123 testfoo0875'
        assert clean_text(sentence) == 'f oo bar testfoo'

    def test_unique_words(self, docs):
        unique_words_result = unique_words(docs)
        words = ['bar', 'baz', 'biz', 'boo', 'foo']
        assert sorted(unique_words_result.keys()) == words
