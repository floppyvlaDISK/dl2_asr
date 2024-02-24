#!/bin/bash


# Problem: transcript file contains incorrect path: missing wav.
# is:        /AUDIOBOOKS/15_річний_капітан/spk_id_10000002_1602308265177.wav
# should be: /AUDIOBOOKS/15_річний_капітан/wav/spk_id_10000002_1602308265177.wav
# result:    Key 'AUDIOBOOKS_15_річний_капітан_wav_spk_id_10000002_1602308265177.wav' does not exist
# so file is moved, but transcript for it is not added to {split}.tsv

# ./fix_missing_transcripts.sh AUDIOBOOKS 15_річний_капітан

if [ -z $1 ]; then
    echo "missing dataset folder."
    exit 1
fi
target_folder=$1
target_folder_path="datasets/nested_datasets/$target_folder"

if [ -z $2 ]; then
    echo "missing subfolder."
    exit 1
fi
subfolder=$2

transcripts_file="$target_folder_path/$subfolder/etc/txt.done.data"
if [[ -e "$transcripts_file" ]]; then
    while read -r audio_path transcript; do
        audio_key=$(echo "${audio_path:1}" | tr '/_' '_')
        if echo "$audio_key" | grep -q "_wav_"; then
            continue
        fi

        audio_key=$(echo "$audio_key" | sed "s/${target_folder}_${subfolder}/${target_folder}_${subfolder}_wav/g")
        audio_path=$(echo $(find datasets/flat_dataset/ -name "$audio_key"))
        if [[ -z $audio_path ]]; then
            continue;
        else
            split=$(echo "$audio_path" | cut -d'/' -f3)
            echo -e "$audio_key\t$transcript" >> "datasets/flat_dataset/${split}.tsv"
        fi
    done < "$transcripts_file"
else
    echo "txt.done.data is missing for $subfolder"
fi
