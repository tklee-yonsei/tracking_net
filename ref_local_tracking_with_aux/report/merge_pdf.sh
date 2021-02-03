#!/bin/bash -eu

#./merge_pdf.sh -f "[input] current image@current image" -f "[input] p1 label@prev label" -t "assembly.pdf"

F_CNT=0
while getopts "f:t:" opt; do
    case ${opt} in
        f)
            FOLDERS+=("$OPTARG")
            F_CNT=$(($F_CNT+1));;
        t)
            TARGETFILE="$OPTARG";;
    esac
done

IFS='@'
read -a F0_ARR <<< "${FOLDERS[0]}"
F0_FOLDER="${F0_ARR[0]}"

CMD="{ "

for FULLFILE in "${F0_FOLDER}"/*.png
do
    FILE_NAME="${FULLFILE##*/}"
    CMD+="montage"
    for FOLDER in "${FOLDERS[@]}"
    do
        read -a FOLDER_ARR <<< "${FOLDER}"
        FOLDER_NAME="${FOLDER_ARR[0]}"
        FOLDER_TAG="${FOLDER_ARR[1]}"

        CMD+=" -label '${FOLDER_TAG}' '${FOLDER_NAME}/${FILE_NAME}'"
    done
    CMD+=" -pointsize 24 -tile ${F_CNT}x1 -geometry +20+20 -border 1 -bordercolor '#444444' miff:- ; "
done

CMD+="} | montage miff:- -tile 1x1 -geometry +1+1 '${TARGETFILE}'"
eval $CMD

