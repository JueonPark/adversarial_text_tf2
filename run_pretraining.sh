#!/bin/bash
shopt -s extglob

for filename in xla_hlo/*; do
  if [[ $filename == *"offline_execution_result"* ]]
  then
    echo $filename
    continue
  else
    rm $filename
  fi
done
rm xla_hlo/*/*

# Basic Configurations for 
PRETRAIN_DIR=./imdb_pretrain
IMDB_DATA_DIR=./dataset/imdb

VOCAB_SIZE=87007
EMBEDDING_DIMS=256
RNN_CELL_SIZE=1024
BATCH=256
EPOCHS=1

# XLA
export TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2"
export TF_DUMP_GRAPH_PREFIX="./xla_hlo"

# simulation
export TRACER_TOOL=/home/jueonpark/cxl-simulator/multi_gpu_simulator/util/tracer_nvbit/tracer_tool/tracer_tool.so
export POST_PROCESSING=/home/jueonpark/cxl-simulator/multi_gpu_simulator/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing

# caution!
export DYNAMIC_KERNEL_LIMIT_START=999998
export DYNAMIC_KERNEL_LIMIT_END=999999

# additional runtime environment variables for tensorflow
# export TF_CPP_MIN_VLOG_LEVEL=1
# export ENABLE_CONSOLE=true

# execution options:
# $1:
# - vanila for no
# - pm for pattern matching
# - fo for fusion offloading
# - pm_fo for both pattern matching & fusion offlaoding
# - ideal for ideal offloading
# - pm_ideal for both pattern matching & ideal offloading
# $2: trace generation
# - keyword "trace" given
# $3: xla_ndpx_use_offline_result
# - 0 for using GPU results
# - 1 for using SIM results
# - on default(no $3 input), use GPU results
if [ $1 = "vanila" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text  --xla_dump_to=./xla_hlo "
elif [ $1 = "pm" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_cudnn_batchnorm=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_optimizer=true --xla_dump_to=./xla_hlo "
elif [ $1 = "fo" ]
then
  if [ $# = 3 ]
  then
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_offline_result=$3 --xla_dump_to=./xla_hlo "
  else
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_offline_result=0 --xla_dump_to=./xla_hlo "
  fi
elif [ $1 = "pm_fo" ]
then
  if [ $# = 3 ]
  then
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_cudnn_batchnorm=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_optimizer=true --xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_offline_result=$3 --xla_dump_to=./xla_hlo "
  else
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_cudnn_batchnorm=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_optimizer=true --xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_offline_result=0 --xla_dump_to=./xla_hlo "
  fi
elif [ $1 = "ideal" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_ideal_offloading --xla_dump_to=./xla_hlo "
elif [ $1 = "pm_ideal" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_cudnn_batchnorm=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_optimizer=true --xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_ideal_offloading --xla_dump_to=./xla_hlo "
else
  echo "flags: vanila, pm, fo, pm_fo, ideal, idea_fo"
	exit 0
fi

# whether to get trace or not
if [ $# -ge 2 ] && [ $2 = "trace" ]
then
  LD_PRELOAD=$TRACER_TOOL python pretrain.py \
    --train_dir=$PRETRAIN_DIR \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=87007 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --num_candidate_samples=1024 \
    --batch_size=256 \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.9999 \
    --max_steps=100000 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings
  $POST_PROCESSING ./traces/kernelslist
else
  pretrain_dir=/tmp/models/imdb_pretrain
  python pretrain.py \
    --train_dir=$PRETRAIN_DIR \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=87007 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --num_candidate_samples=1024 \
    --batch_size=256 \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.9999 \
    --max_steps=1 \
    --max_grad_norm=1.0 \
    --num_timesteps=1 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings
fi
