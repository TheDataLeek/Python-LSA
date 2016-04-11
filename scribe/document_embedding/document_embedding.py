#!/usr/bin/env python3.5

"""
Analyze embedding of documents in R^n space using paragraph vector algorithm
"""

import sys
import argparse
from ..LSA.LSA import open_documents


def main():
    filename = '../data/hillary/cleanhillary.txt'
    docs = open_documents(filename, 10)
    print(docs)


if __name__ == '__main__':
    sys.exit(main())
