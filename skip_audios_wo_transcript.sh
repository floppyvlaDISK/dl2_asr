#!/bin/bash

logfile="./error.log"
backlog_folder="datasets/unlabeled_speech_ds/leftovers_from_nested"

if [[ -e "$logfile" ]]; then
    while IFS= read -r line; do
        if [[ "${line:0:5}" == "Key '" ]]; then
            audio_file=$(echo "$line" | grep -o "'.*'" | sed "s/'//g")
            audio_path=$(echo $(find datasets/flat_dataset/ -name "$audio_file"))
            if [[ -z $audio_path ]]; then
                echo "failed to find $audio_file"
                continue;
            else
                split=$(echo "$audio_path" | cut -d'/' -f3)

                has_transcript=$(grep "$audio_file" "datasets/flat_dataset/$split.tsv")
                if ! [[ -z $has_transcript ]]; then
                    echo "has transcript"
                    continue
                fi

                mkdir -p "$backlog_folder/$split"
                if ! mv "$audio_path" "$backlog_folder/$split"; then
                    echo "Error moving the file $old_path to $new_path"
                fi
            fi
        fi
    done < "$logfile"
else
    echo "$logfile is missing"
    continue
fi
