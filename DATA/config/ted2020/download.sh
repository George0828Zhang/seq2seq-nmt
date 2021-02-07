#!/usr/bin/env bash
URLS=(
    "http://opus.nlpl.eu/download.php?f=TED2020/v1/moses/en-zh_tw.txt.zip"
)
FILES=(
    "en-zh_tw.txt.zip"
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
    
    unzip $file
done

rename s/_tw//g TED2020*