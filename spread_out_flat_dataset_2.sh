#!/bin/bash

split=$1
if [ -z "$split" ]; then
    echo "split is missing"
    exit 1
fi

target_dir="datasets/flat_dataset/$split"
target_tsv="datasets/flat_dataset/$split.tsv"
count=$(find $target_dir -name "*.wav" | wc -l)
readarray -t audios < <(find "$target_dir" -name "*.wav")
chunks=200
batch_size=$(echo "scale=2; $count / $chunks" | bc)
batch_size=$(echo "scale=0; ($batch_size + 0.5) / 1" | bc) # floor
echo "batch size: $batch_size"
echo "count: $count"

for ((i = 50; i < 100; i++)); do
    echo "i: $i"
    start_item=$(echo "scale=2; $i * $batch_size" | bc)
    start_item=$(echo "scale=0; $start_item / 1" | bc)

    end_item=$(echo "scale=2; $start_item + $batch_size - 1" | bc)
    end_item=$(echo "scale=0; $end_item / 1" | bc)
    if ((end_item > count)); then
        end_item=$count
    fi

    chunk_name="chunk_$((i + 1))"
    chunk_tsv_file="datasets/spread_flat_dataset/$chunk_name/$split.tsv"
    chunk_data_dir="datasets/spread_flat_dataset/$chunk_name/$split"
    mkdir -p "$chunk_data_dir"
    if [ ! -e "$chunk_tsv_file" ]; then
        touch "$chunk_tsv_file"
    fi

    for ((j = start_item; j < end_item; j++)); do
        audio_full_path=${audios[$j]}

        # copy audio file
        cp "$audio_full_path" "$chunk_data_dir"

        # copy transcript line correspoding to audio file
        audio_name=$(basename "$audio_full_path")
        audio_name=${audio_name/\[/\\[}
        tsv_line=$(grep "^$audio_name" "$target_tsv" | head -n 1)
        if [ -z "$tsv_line" ]; then
            echo "tsv_line missing for $audio_name"
        else
            echo -e "$tsv_line" >> "$chunk_tsv_file"
        fi
    done
done
