#!/bin/bash

TEST_ID="$1"
TARGET_FOLDER="$2"
RESULT_FOLDER=$TARGET_FOLDER"/"$TEST_ID

RMDIR_CMD="rm -r "
RMDIR_CMD+="'"$RESULT_FOLDER"'"
eval $RMDIR_CMD

MKDIR_CMD="mkdir -p "
MKDIR_CMD+="'"$RESULT_FOLDER"'"
eval $MKDIR_CMD


GET_RESULT_CMD="gsutil -m -o 'GSUtil:parallel_process_count=1' cp -r "
GET_RESULT_CMD+="gs://cell_dataset/data/"$TEST_ID" "
GET_RESULT_CMD+="'"$RESULT_FOLDER"'"
eval $GET_RESULT_CMD

RESULT_MV_CMD="mv "
RESULT_MV_CMD+="'"$RESULT_FOLDER"/"$TEST_ID"' "
RESULT_MV_CMD+="'"$RESULT_FOLDER"/test_result'"
eval $RESULT_MV_CMD
