#!/usr/bin/env bash
# setup cache dir for raw,
export RAW=$PREFIX/raw
export DATASET="ted2020"
export CACHE=$PREFIX/$DATASET
export OUTDIR=$DATABIN/$DATASET

export SPM_MODEL="Current"
export BINARIZEARGS="--joined-dictionary --bpe sentencepiece"