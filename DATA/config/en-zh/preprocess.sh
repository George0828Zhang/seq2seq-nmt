#!/usr/bin/env bash
src=$SRCLANG
tgt=$TGTLANG
raw=$RAW
prep=$CACHE/prep
ready=$CACHE/ready
lang=en-zh
jieba=jieba_fast

mkdir -p $prep $ready

VALID_SETS=(
    # "IWSLT17.TED.dev2010.en-zh"
    # "IWSLT17.TED.tst2010.en-zh"
    # "IWSLT17.TED.tst2011.en-zh"
    # "IWSLT17.TED.tst2012.en-zh"
    "IWSLT17.TED.tst2013.en-zh"
    "IWSLT17.TED.tst2014.en-zh"
    "IWSLT17.TED.tst2015.en-zh"
)

cd $PREFIX
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

SCRIPTS=$(pwd)/mosesdecoder/scripts
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

cd $raw

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $raw/$lang/$f \
    | grep -v '<url>' \
    | grep -v '<talkid>' \
    | grep -v '<keywords>' \
    | grep -v '<speaker>' \
    | grep -v '<reviewer' \
    | grep -v '<translator' \
    | grep -v '<doc' \
    | grep -v '</doc>' \
    | sed -e 's/<title>//g' \
    | sed -e 's/<\/title>//g' \
    | sed -e 's/<description>//g' \
    | sed -e 's/<\/description>//g' \
    | sed 's/^\s*//g' \
    | sed 's/\s*$//g' \
    > $raw/$f

    if [[ $l == "zh" ]]; then            
        cat $raw/$f | \
            $REM_NON_PRINT_CHAR | \
            python -m $jieba -d > $prep/train.dirty.$l
    else
        cat $raw/$f | \
            $REM_NON_PRINT_CHAR > $prep/train.dirty.$l
    fi
    echo ""
done
perl $CLEAN -ratio 9 $prep/train.dirty $src $tgt $prep/train 1 1000

echo "pre-processing valid data..."
for l in $src $tgt; do
    echo -n "" > $prep/valid.$l # reset to blank file
    for fname in ${VALID_SETS[@]}; do
        o=$raw/$lang/$fname.$l.xml
        f=$raw/$fname.$l

        echo $o $f
        grep '<seg id' $o | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\’/\'/g" > $f

        if [[ $l == "zh" ]]; then
            cat $f | \
                $REM_NON_PRINT_CHAR | \
                python -m $jieba -d >> $prep/valid.$l
        else
            cat $f | \
                $REM_NON_PRINT_CHAR >> $prep/valid.$l
        fi

        echo ""
    done
done