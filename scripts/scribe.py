#!/usr/bin/env python3.5

import sys
import os
import argparse
import logging
import enforce
import time
import numpy as np
import scipy.sparse.linalg as ssl

sys.path.insert(0, os.path.abspath('..'))
from scribe.LSA import LSA


def main():
    logging.basicConfig(filename='../logs/LSA.log', level=logging.DEBUG)

    logging.info('Program Start')
    """ Manage Execution """
    args = get_args()

    logging.info('Program Arguments: {}'.format(str(args)))

    documents, doccount = LSA.open_documents(args.filename, args.count)
    print('Program Start. Loaded Data. Time Elapsed: {}\n'.format(time.clock()))
    logging.info('Loaded Data. Time Elapsed: {}'.format(time.clock()))

    words = LSA.get_unique_words(documents, args.workers)
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

    docmatrix, documents = LSA.get_sparse_matrix(documents, words, args.workers)
    print('Calculated Sparse Matrix\nTime Elapsed: {}\n'.format(time.clock()))
    logging.info('Calculated Sparse Matrix. Time Elapsed: {}'.format(time.clock()))

    u, s, vt = ssl.svds(docmatrix.T, k=args.svdk)
    print('Calculated SVD Decomposition\nTime Elapsed: {}'.format(time.clock()))
    logging.info('Calculated SVD Decomposition. Time Elapsed: {}'.format(time.clock()))

    if args.save:
        output = {'u':u, 'd': np.diag(s), 'vt':vt,
                  'documents': np.array(documents, dtype=object),
                  'words': np.array(list(sorted(words.keys())), dtype=object)}
        print('Saving U: {}, S: {}, V.T: {}'.format(u.shape, s.shape, vt.shape))
        LSA.save_output(output)


@enforce.runtime_validation
def get_args() -> argparse.Namespace:
    """
    Get Command line Arguments

    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workers', type=int, default=32,
                        help=('Number of workers to use for multiprocessing'))
    parser.add_argument('-c', '--count', type=int, default=-1,
                        help=('Number of documents to use from original set'))
    parser.add_argument('-k', '--svdk', type=int, default=20,
                        help=('SVD Degree'))
    parser.add_argument('-f', '--filename', type=str, default='../data/jeopardy/jeopardy.csv',
                        help=('File to use for analysis'))
    parser.add_argument('-s', '--save', action='store_true', default=False,
                        help=('Save output in .mat file.'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sys.exit(main())
