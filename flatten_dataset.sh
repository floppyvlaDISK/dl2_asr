#!/bin/bash

# Input path
if [ -z $1 ]; then
    echo "missing dataset folder."
    exit 1
fi
target_folder=$1
target_folder_path="datasets/nested_datasets/$target_folder"

subfolders=$(ls $target_folder_path)
subfolders_count=$(ls $target_folder_path | wc -l)
subfolders_counter=1
for subfolder in $subfolders; do
    echo "$subfolders_counter of $subfolders_count"
    ((subfolders_counter++))

    audio_files=$(ls "$target_folder_path/$subfolder/wav")
    audio_files_count=$(ls "$target_folder_path/$subfolder/wav" | wc -l)

    counter_start_dev=$((audio_files_count * 90 / 100))
    counter_start_dev=$(printf "%.0f" "$counter_start_dev")

    counter_start_test=$((audio_files_count * 80 / 100))
    counter_start_test=$(printf "%.0f" "$counter_start_test")

    transcripts_file="$target_folder_path/$subfolder/etc/txt.done.data"
    declare -A transcript_hash_table
    if [[ -e "$transcripts_file" ]]; then
        while read -r audio_path transcript; do
            audio_key=$(echo "${audio_path:1}" | tr '/_' '_')
            transcript_hash_table["$audio_key"]="$transcript"
        done < "$transcripts_file"
    else
        echo "txt.done.data is missing for $subfolder"
    fi

    counter=0
    for audio_file in $audio_files; do
        old_path="$target_folder_path/$subfolder/wav/$audio_file"

        split="train"
        if [ "$counter" -ge "$counter_start_dev" ]; then
            split="dev"
        elif [ "$counter" -ge "$counter_start_test" ]; then
            split="test"
        fi
        new_audio_file=$(echo "$target_folder/$subfolder/wav/$audio_file" | tr '/_' '_')
        new_path="datasets/flat_dataset/$split/$new_audio_file"
        ((counter++))

        mkdir -p "$(dirname "$new_path")"

        if ! mv "$old_path" "$new_path"; then
            echo "Error moving the file $old_path to $new_path"
        fi

        if [[ -n "${transcript_hash_table[$new_audio_file]}" ]]; then
            echo -e "$new_audio_file\t${transcript_hash_table[$new_audio_file]}" >> "datasets/flat_dataset/${split}.tsv"
        else
            echo "Key '$new_audio_file' does not exist"
        fi
    done
done
