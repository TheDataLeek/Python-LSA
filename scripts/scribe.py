#!/usr/bin/env python3.5

import sys
import os
import argparse
import logging
import enforce

sys.path.insert(0, os.path.abspath('..'))
from scribe.LSA import LSA
from scribe.document_embedding import document_embedding


def main():
    logging.basicConfig(filename='../logs/LSA.log', level=logging.DEBUG)
    logging.info('Program Start')
    """ Manage Execution """
    args = get_args()
    print(args)
    logging.info('Program Arguments: {}'.format(str(args)))

    if args.embedding:
        document_embedding.embedding(args.filename, args.workers)

    if args.lsa:
        LSA.analyze(args.filename, args.workers, args.count, args.svdk, args.save)


@enforce.runtime_validation
def get_args() -> argparse.Namespace:
    """
    Get Command line Arguments
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
    parser.add_argument('-emb', '--embedding', action='store_true', default=False,
                        help='Perform Document embedding and associated analysis')
    parser.add_argument('-lsa', '--lsa', action='store_true', default=False,
                        help='Perform Latent Sentiment Analysis')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sys.exit(main())
