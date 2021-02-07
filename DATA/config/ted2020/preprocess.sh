#!/usr/bin/env bash
src=$SRCLANG
tgt=$TGTLANG
raw=$RAW
prep=$CACHE/prep
ready=$CACHE/ready
lang=en-zh
# jieba=jieba_fast

mkdir -p $prep $ready

# # \u2581 is for sentencepiece. we use \u2582 instead
# MYDELIMETER='\u2582'

cd $PREFIX
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

SCRIPTS=$(pwd)/mosesdecoder/scripts
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

cd $raw

echo "pre-processing train data..."
for l in $src $tgt; do
    f=TED2020.$lang.$l
    tok=train.tags.$lang.tok.$l

    # cat $raw/$f \
    # | sed 's/\([A-Z][a-z]*\)//g' \
    # | sed 's/\s*$//g' \
    # | sed 's/^\s*//g' \
    # | sed 's/\s*$//g' \
    # > $raw/$f

    
    cat $raw/$f \
    | perl -pe 's/\(.*?\)//g' \
    | perl -pe 's/（.*?）//g' \
    | sed 's/^\s*//g' \
    | sed 's/\s*$//g' \
    | $REM_NON_PRINT_CHAR > $prep/train-valid.dirty.$l

    # if [[ $l == "zh" ]]; then            
    #     cat $raw/$f | \
    #         $REM_NON_PRINT_CHAR | \
    #         sed "s/ /$(echo -ne ${MYDELIMETER})/g" | \
    #         python -m $jieba -d > $prep/train.dirty.$l
    # else
    #     cat $raw/$f | \
    #         $REM_NON_PRINT_CHAR > $prep/train.dirty.$l
    # fi
    echo ""
done
perl $CLEAN -ratio 9 $prep/train-valid.dirty $src $tgt $prep/train-valid 1 1000

echo "creating train, valid ..."
for l in $src $tgt; do
    awk '{if (NR%99 == 0)  print $0; }' $prep/train-valid.$l > $prep/valid.$l
    awk '{if (NR%99 != 0)  print $0; }' $prep/train-valid.$l > $prep/train.$l
done