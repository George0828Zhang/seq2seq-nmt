#!/usr/bin/env bash
# config
DATA=./DATA
DATA=$(realpath ${DATA})
export PREFIX=${DATA}/rawdata
export DATABIN=${DATA}/data-bin
export CONFIG=${DATA}/config/en-zh
export SRCLANG=en
export TGTLANG=zh
export WORKERS=8
export BPE_TOKENS=8000

# setup path
source $CONFIG/path.sh

# check
if [[ "$1" == "clean" ]]; then
    rm -rf $CACHE $OUTDIR
elif [[ -d $OUTDIR ]]; then
    echo "$OUTDIR already exists. Please change the OUTDIR or remove $OUTDIR"
    exit 1
fi

# download data
bash $CONFIG/download.sh

# preprocess data
bash $CONFIG/preprocess.sh

# (train, )apply spm/bpe
bash learn_or_apply_bpe.sh

# binarize
mkdir -p $OUTDIR
LOCS=""
for split in train valid; do
    if [[ -f $CACHE/ready/${split}.$SRCLANG ]]; then
        LOCS="$LOCS --${split}pref $CACHE/ready/${split}"
    fi
done
python -m fairseq_cli.preprocess \
    --source-lang $SRCLANG \
    --target-lang $TGTLANG \
    $LOCS \
    --destdir $OUTDIR \
    --workers $WORKERS \
    $BINARIZEARGS