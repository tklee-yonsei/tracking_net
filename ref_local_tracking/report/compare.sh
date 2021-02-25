#!/bin/bash

folder1=/Users/Mark/tmp/1
folder2=/Users/Mark/tmp/2
differences=/Users/Mark/tmp/diff

while getopts "f:t:s:" opt; do
    case ${opt} in
        f)
            FROM_FOLDER="$OPTARG";;
        t)
            WITH_FOLDER="$OPTARG";;
        s)
            TARGET_FOLDER=$OPTARG;;
    esac
done

for FULLFILE in "${FROM_FOLDER}"/*.png
do
    FILE_NAME="${FULLFILE##*/}"
    echo $FILE_NAME
    compare "$FROM_FOLDER/$FILE_NAME" "$WITH_FOLDER/$FILE_NAME" "$TARGET_FOLDER/$FILE_NAME"
done
