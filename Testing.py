
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
import scipy.sparse.linalg as ssl
import scipy.sparse as scs
from tqdm import tqdm


# In[2]:

get_ipython().magic('load_ext cython')


# First we'll load in the data. Using the Jeapardy data as it's small.

# In[3]:

data = pd.read_csv('./data/JEOPARDY_CSV.csv')
data = data[:1000]
data.head()


# In[29]:

data.values[0]


# In[4]:

m = len(data)


# In[5]:

get_ipython().run_cell_magic('cython', '', "\nimport numpy as np\ncimport numpy as np\nimport scipy.sparse as scs\nfrom scipy.sparse import dok_matrix\n\nalphabet = 'abcdefghijklmnopqrstuvwxyz'\n\ndef unique_words(list sentences):\n    cdef dict words = {}\n    cdef int n = len(sentences)\n    cdef int i, j\n    for i in range(n):\n        sent_list = [w.lower() for w in sentences[i].split(' ')]\n        clean_sent_list = []\n        for j in range(len(sent_list)):\n            newword = ''\n            for char in sent_list[j]:\n                if char in alphabet:\n                    newword += char\n            clean_sent_list.append(newword)\n        for word in clean_sent_list:\n            if word != '':\n                try:\n                    words[word] += 1\n                except KeyError:\n                    words[word] = 1\n    wordlist = sorted(words.keys())\n    return wordlist, len(wordlist), words\n\n# Use tf-idf\n# https://en.wikipedia.org/wiki/Tf%E2%80%93idf\ndef populate_doc_matrix(docmatrix, wordlist, word_freq, np.ndarray data):\n    cdef int n = len(data)   # number of documents\n    cdef int i, j, k, m\n    # construct word index first\n    # This tells us (for any word) what index it is in in document\n    print('Constructing Word Reference')\n    wordref = {}\n    for i in range(len(wordlist)):\n        wordref[wordlist[i]] = i\n    # Now populate sparse matrix\n    print('Populating Sparse Matrix')\n    for i in range(n):\n        for j in range(2):\n            words = [w.lower() for w in data[i, j].split(' ') if w != '']\n            m = len(words)\n            for k in range(m):\n                word = words[k]\n                cword = ''\n                for char in word:\n                    if char in alphabet:\n                        cword += char\n                if cword != '':\n                    docmatrix[i, wordref[cword]] += 1\n    # finish weighting\n    print('Weighting Matrix')\n    m, n = docmatrix.shape\n    weighted_docmatrix = dok_matrix((m, n), dtype=float)\n    for i in range(n):\n        weighted_docmatrix[:, i] = docmatrix[:, i] * np.log(m / word_freq[wordlist[i]])\n    return weighted_docmatrix, wordref")


# In[6]:

words, n, wordfreq = unique_words(list(np.concatenate((data[[' Question']].values[:, 0],
                                  data[[' Answer']].values[:, 0]))))


# In[15]:

print('{} Documents (m) by {} Unique Words (n)\n\nTop 100 Most Frequent Words:{}'.format(
        m, n, ','.join([tup[0] for tup in sorted(wordfreq.items(), key=lambda tup: -tup[1])[:100]])))


# In[8]:

docmatrix = dok_matrix((m, n), dtype=float)   # m-docs, n-unique words


# In[9]:

ndocterm, wordref = populate_doc_matrix(docmatrix, words, wordfreq,
                                data[[' Question', ' Answer']].values)


# In[10]:

ndocterm


# In[38]:

u, s, vt = ssl.svds(ndocterm.T, k=20)
u.shape, s.shape, vt.shape


# In[37]:

np.save('umatrix.npy', u)
np.save('smatrix.npy', s)
np.save('vtmatrix.npy', vt)


# Now that we have our $k$th-order decomposition, let's query the word "Species".

# In[187]:

wordref['species']


# In[34]:

get_ipython().system('ls')


# In[35]:

np.load('./umatrix.npy')


# In[ ]:



