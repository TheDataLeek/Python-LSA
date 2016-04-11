import numpy as np
import scipy.io as scio
import sys

mat = scio.loadmat('output.mat')
u = mat['u']
vt = mat['vt']
d = mat['d']
wordlist = mat['words']
documents = mat['documents']

num_docs = vt.shape[1]
num_words = u.shape[0]

word_mat = np.dot(u,d)
doc_mat = np.dot(d,vt)

query = input('Enter the query word \n')
query_flag = 0

for i in range(0,num_words):
    if query.lower() == wordlist[0,i]:
        word_index = i
        query_flag = 1

if query_flag == 0:
    sys.exit("Enter a valid word!")
q = word_mat[word_index,:]

rank = {}
for i in range(0,num_docs):
    rank[i] = np.dot(doc_mat[:,i],q)/(np.linalg.norm(doc_mat[:,i])*np.linalg.norm(q))

r = sorted(range(len(rank)), key=lambda k: rank[k])
print('\n')
print(documents[0,r[999]])
print(documents[0,r[998]])
