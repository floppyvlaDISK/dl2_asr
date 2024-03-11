#!/bin/bash

split=$1
if [ -z "$split" ]; then
    echo "split is missing"
    exit 1
fi

target_dir="datasets/flat_dataset/$split"
target_tsv="datasets/flat_dataset/$split.tsv"

audios_to_fix=(
)

length=${#audios_to_fix[@]}
for ((i = 0; i < length; i++)); do
    audio_name="${audios_to_fix[$i]}"
    audio_name=${audio_name/\[/\\[}

    audio_full_path=$(find "datasets/spread_flat_dataset" -name "$audio_name")
    chunk_tsv_file="$(dirname "$audio_full_path").tsv"
    echo "chunk_tsv_file: $chunk_tsv_file"

    tsv_line=$(grep "^$audio_name" "$target_tsv" | head -n 1)
    if [ -z "$tsv_line" ]; then
        echo "still no tsv_line for $audio_name"
    else
        echo -e "$tsv_line" >> "$chunk_tsv_file"
    fi
done
