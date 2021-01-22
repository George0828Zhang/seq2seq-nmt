#!/usr/bin/env bash
NAME=baseline
SRC=en
TGT=zh
DATA=DATA/data-bin/en-zh
logdir=exp/logdir/${NAME}
savedir=exp/checkpoints/${NAME}

export CUDA_VISIBLE_DEVICES=1 # use cuda:0
FP16=false # half precision for faster training 

fairseq-hydra-train \
    task.data=${DATA} \
    task.source_lang=${SRC} \
    task.target_lang=${TGT} \
    common.fp16=${FP16} \
    common.user_dir='./extensions/' \
    common.tensorboard_logdir=${logdir} \
    checkpoint.save_dir=${savedir} \
    --config-dir './config/' \
    --config-name 'baseline' # use baseline.yaml