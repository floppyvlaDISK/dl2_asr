#!/bin/bash

if [ -z $1 ]; then
    echo "missing dataset folder."
    exit 1
fi
target_folder=$1
target_folder_path="datasets/nested_datasets/$target_folder"

subfolders=("$target_folder_path"/*)
subfolders_count=$(ls $target_folder_path | wc -l)
subfolders_counter=1
for subfolder_path in "${subfolders[@]}"; do
    if ! [ -d "$subfolder_path" ]; then
        echo "excluding $subfolder_path due to -d check"
        continue
    fi
    subfolder=$(basename "$subfolder_path")
    if ! [ -z "$2" ] && [ "$subfolder" != "$2" ]; then
        echo "excluding $subfolder due to $2"
        continue
    fi

    echo "$subfolders_counter of $subfolders_count"
    ((subfolders_counter++))

    audio_files=$(ls "$target_folder_path/$subfolder/wav")
    audio_files_count=$(ls "$target_folder_path/$subfolder/wav" | wc -l)
    if [[ "$audio_files_count" -eq 0 ]]; then
        continue
    fi

    counter_start_dev=$((audio_files_count * 90 / 100))
    counter_start_dev=$(printf "%.0f" "$counter_start_dev")

    counter_start_test=$((audio_files_count * 80 / 100))
    counter_start_test=$(printf "%.0f" "$counter_start_test")

    transcripts_file="$target_folder_path/$subfolder/etc/txt.done.data"
    declare -A transcript_hash_table
    if [[ -e "$transcripts_file" ]]; then
        while IFS= read -r line; do
            # Use sed to split the string at ".wav"
            audio_path=$(echo "$line" | sed 's/\(.*\.wav\).*/\1/')
            audio_key=$(echo "${audio_path:1}" | tr '/_' '_')
            if [ -z $audio_key ]; then
                continue
            fi

            transcript=$(echo "$line" | sed 's/.*\.wav\(.*\)/\1/')
            # trim only leading whitespaces
            transcript="${transcript#"${transcript%%[![:space:]]*}"}"

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
