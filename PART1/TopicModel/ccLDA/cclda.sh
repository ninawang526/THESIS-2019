#!/bin/sh

cd TAM_ccLDA

java LearnTopicModel -model cclda -input ../data/cclda.txt -iters 500 -Z 30

python topwords_cclda.py ../data/cclda.txt.assign > ../data/output_topwords_cclda.txt