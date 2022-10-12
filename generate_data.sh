#!/bin/bash
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz \
    -O /tmp/imdb.tar.gz
tar -xf /tmp/imdb.tar.gz -C /tmp

IMDB_DATA_DIR=/tmp/imdb
python gen_vocab.py \
    --output_dir=$IMDB_DATA_DIR \
    --dataset=imdb \
    --imdb_input_dir=/tmp/aclImdb \
    --lowercase=False

python gen_data.py \
    --output_dir=$IMDB_DATA_DIR \
    --dataset=imdb \
    --imdb_input_dir=/tmp/aclImdb \
    --lowercase=False \
    --label_gain=False
