## Setup

1. Install fairseq
```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git commit -m 9a1c497
pip install .
```
2. (Optional) Install apex for faster fp16 training
```bash
# note that apex requires that nvcc on system should be same version as that used to build torch 
# e.g. torch 1.7.0+cu110 ==> nvcc -V should be 11.0,
#      torch 1.6.0+cu102 ==> nvcc -V should be 10.2, etc.

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
**Download & preprocess dataset for training:**

1. Modify `get_data.sh` to suit your need
```bash
DATA=./DATA
export PREFIX=${DATA}/rawdata       # path to cache raw data 
export DATABIN=${DATA}/data-bin     # path to output data
export CONFIG=${DATA}/config/en-zh  # path to config for dataset, including {download, preprocess, path}.sh
export SRCLANG=en                   # source language
export TGTLANG=zh                   # source language
export WORKERS=8                    # workers / threads
export BPE_TOKENS=8000              # desired bpe tokens or spm pieces. only used if path.sh specify 'Current' (i.e. to learn bpe)
```
2. Run
```bash
bash get_data.sh
```

**Train**
```bash
# optionally tune hyper parameters in exp/config/*.yaml
# then begin training
bash hydra_train.sh
```