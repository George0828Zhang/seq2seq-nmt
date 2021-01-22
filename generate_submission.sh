#!/usr/bin/env bash
NAME=baseline
INPUT=testdata/test.en
CHECKDIR=exp/checkpoints/${NAME}
AVG=true
OUTDIR=result

DATA=DATA/data-bin/en-zh
SRC=en
TGT=zh

export CUDA_VISIBLE_DEVICES=1
mkdir -p ${OUTDIR}

if [[ $AVG == "true" ]]; then
  CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
  python ./scripts/average_checkpoints.py \
    --inputs ${CHECKDIR} --num-epoch-checkpoints 10 \
    --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
  CHECKPOINT_FILENAME=checkpoint_best.pt
fi

# preprocess input sentences
SPM_MODEL=${DATA}/spm.model
cat ${INPUT} \
  | python scripts/spm_encode.py --model ${SPM_MODEL} \
  > ${OUTDIR}/input.${SRC}.tok

# translate using trained model
cat ${OUTDIR}/input.${SRC}.tok \
  | python -m fairseq_cli.interactive ${DATA} \
    --user-dir './extensions/' \
    --task translation \
    --source-lang ${SRC} --target-lang ${TGT} \
    --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
    --buffer-size 2000 --batch-size 128 \
    --beam 5 --remove-bpe=sentencepiece \
  > ${OUTDIR}/generate.log

# extract system output from log
grep ^H ${OUTDIR}/generate.log | cut -f3 \
  | perl scripts/normalize-punctuation.perl ${TGT} > ${OUTDIR}/output.${TGT}.tok

# 'normalize-punctuation' and 'tokenizeChinese' will be performed by server instead!
# # split to char
# python scripts/tokenizeChinese.py ${OUTDIR}/output.${TGT}.tok ${OUTDIR}/output.${TGT}.char
