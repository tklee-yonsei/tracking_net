#!/bin/bash

TRAINING_ID="$1"
TARGET_FOLDER="$2"
RESULT_FOLDER=$TARGET_FOLDER"/"$TRAINING_ID

RMDIR_CMD="rm -r "
RMDIR_CMD+="'"$RESULT_FOLDER"'"
eval $RMDIR_CMD

MKDIR_CMD="mkdir -p "
MKDIR_CMD+="'"$RESULT_FOLDER"'"
eval $MKDIR_CMD


GET_RESULT_CMD="gsutil -m -o 'GSUtil:parallel_process_count=1' cp -r "
GET_RESULT_CMD+="gs://cell_dataset/data/"$TRAINING_ID" "
GET_RESULT_CMD+="'"$RESULT_FOLDER"'"
eval $GET_RESULT_CMD

RESULT_MV_CMD="mv "
RESULT_MV_CMD+="'"$RESULT_FOLDER"/"$TRAINING_ID"' "
RESULT_MV_CMD+="'"$RESULT_FOLDER"/training_result'"
eval $RESULT_MV_CMD


MKDIR_TF_LOG_CMD="mkdir -p "
MKDIR_TF_LOG_CMD+="'"$RESULT_FOLDER"/tf_log'"
eval $MKDIR_TF_LOG_CMD

GET_TF_LOG_CMD="gsutil -m -o 'GSUtil:parallel_process_count=1' cp -r "
GET_TF_LOG_CMD+="gs://cell_dataset/save/tf_logs/"$TRAINING_ID" "
GET_TF_LOG_CMD+="'"$RESULT_FOLDER"/tf_log'"
eval $GET_TF_LOG_CMD
