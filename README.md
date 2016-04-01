# Latent Semantic Analysis in Python

In this project we will perform [latent semantic
analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) of large
document sets.

We first create a [document term
matrix](https://en.wikipedia.org/wiki/Document-term_matrix), and then perform
[SVD decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition).

This document term matrix uses
[tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) weighting.


Notes to @rrish:
    * This actually does work for the entire jeopardy dataset, with all 200,000
      documents and 100,000 unique words. Warning, if you do run it on that, it
      needs about 2GB of memory to store everything, so be careful.
    * The global `WORKERS` variable sets how many worker processes to create.
      Feel free to play around for performance. (I haven't yet)
    * In terms of timing, as it stands it can analyze all 200,000 documents and
      create the document-term matrix in about 45-50 seconds on my machine
      (mileage may vary based on cores/etc.)
    * It is currently using the basic tf-idf weighting. We may wish to adjust
      this later.
    * Email me with questions. I'll try to get this thing commented ASAP.
