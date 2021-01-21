#!/usr/bin/env bash
URLS=(
    "https://onedrive.live.com/download?cid=3E549F3B24B238B4&resid=3E549F3B24B238B4%214968&authkey=ABBzIYQ62G5Vh4w"
)
FILES=(
    "en-zh.tgz"
)

mkdir -p $RAW

echo "Downloading data ..."
cd $RAW
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget -O $file "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
    fi
    
    tar zxvf $file
done