#!/usr/bin/env bash

# This script normalizes the punctuation by running the following scripts
# 'normalize-punctuation.perl' & 'replace-unicode-punctuation.perl'
# Both are from this repository
# https://github.com/moses-smt/mosesdecoder.git

REF=test.real.zh
SCRIPTS=mosesdecoder/scripts/tokenizer
PRIVATE=""

while test -n "$1"; do
  case "$1" in
    -v|--private)
        PRIVATE='true'
        shift 1
        ;;
    *)
        break
        ;;
  esac
done
HYP=$1

if [ ! -d ${SCRIPTS} ] || [ ! -f ${REF} ] || [ ! -f ${HYP} ]; then
    echo 0.0
    exit
fi

normalize () {
  N=2000
  if [ -z "$PRIVATE" ]; then tail -n +$(($N+1)) $1; else head -n $N $1; fi |
  perl $SCRIPTS/replace-unicode-punctuation.perl |
  perl $SCRIPTS/normalize-punctuation.perl -l zh
}

normalize $REF > ${REF%.zh}.norm.zh

# sacrebleu
normalize $HYP | \
python -m sacrebleu \
--tokenize zh \
--score-only \
--width 2 \
--force \
${REF%.zh}.norm.zh