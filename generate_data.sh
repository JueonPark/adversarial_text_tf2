#!/bin/bash
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz \
    -O ./dataset/imdb.tar.gz
tar -xf ./dataset/imdb.tar.gz -C ./dataset

IMDB_DATA_DIR=./dataset/imdb
python gen_vocab.py \
    --output_dir=$IMDB_DATA_DIR \
    --dataset=imdb \
    --imdb_input_dir=./dataset/aclImdb \
    --lowercase=False

python gen_data.py \
    --output_dir=$IMDB_DATA_DIR \
    --dataset=imdb \
    --imdb_input_dir=./dataset/aclImdb \
    --lowercase=False \
    --label_gain=False
