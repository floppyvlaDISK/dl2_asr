#!/bin/bash

split=$1
if [ -z "$split" ]; then
    echo "split is missing"
    exit 1
fi

chunks=("datasets/spread_flat_dataset"/*)
chunks_len=${#chunks[@]}

if [ ! "200" -eq "$chunks_len" ]; then
    echo "MISSMATCH: chunks length = $chunks_len"
    for ((i = 0; i < 200; i++)); do
        chunk_name="chunk_$((i + 1))"
        if [ ! -d "datasets/spread_flat_dataset/$chunk_name" ]; then
            echo "MISSING: $chunk_name"
        fi
    done
    exit 1
fi

for chunk_path in "${chunks[@]}"; do
    dest="$chunk_path/$split"
    file_count=$(find "$dest/" -name "*.wav" | wc -l)
    tsv_length=$(wc -l < "$dest.tsv")
    if [ $file_count -eq $tsv_length ]; then
        echo "$file_count = $tsv_length"
    else
        echo "MISSMATCH for $dest: $file_count != $tsv_length"
    fi
done
