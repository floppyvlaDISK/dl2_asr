#!/bin/bash


target_folder_path="datasets/nested_datasets/$1"
subfolder="$2"
transcripts_file="$target_folder_path/$subfolder/etc/txt.done.data"

if [[ -e "$transcripts_file" ]]; then
    while IFS= read -r line; do
        # Use sed to split the string at ".wav"
        audio_path=$(echo "$line" | sed 's/\(.*\.wav\).*/\1/')
        audio_key=$(echo "${audio_path:1}" | tr '/_' '_')

        transcript=$(echo "$line" | sed 's/.*\.wav\(.*\)/\1/')
        transcript=${transcript:1}

        audio_path=$(echo $(find datasets/flat_dataset/ -name "$audio_key"))
        if [[ -z $audio_path ]]; then
            echo "failed to find $audio_key"
            continue;
        else
            split=$(echo "$audio_path" | cut -d'/' -f3)
            echo -e "$audio_key\t$transcript" >> "datasets/flat_dataset/${split}.tsv"
        fi

    done < "$transcripts_file"
else
    echo "txt.done.data is missing for $subfolder"
fi
