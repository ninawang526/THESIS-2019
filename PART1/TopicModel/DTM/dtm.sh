#!/bin/sh

cd dtm/dtm

./main \
  --ntopics=20 \
  --mode=fit \
  --rng_seed=0 \
  --initialize_lda=true \
  --corpus_prefix=../../data/doc \
  --outname=../../data/model_run \
  --top_chain_var=0.5 \
  --alpha=0.01 \
  --lda_sequence_min_iter=6 \
  --lda_sequence_max_iter=20 \
  --lda_max_em_iter=10

cd ../..