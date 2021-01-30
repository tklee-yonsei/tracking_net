# Report

## 1. Merge Images

`merge_pdf.sh`

- Options
  - `-f`: "[Folder Path]@[Label on Image]"
    - Multiple `-f` options can be used. 
    - The order of image placement is in the order of `-f`.
    - Lists the file names that exist in the folder of the first `-f` option applied to the folder of the later `-f` options.
    - If the matching file does not exist in later folders, an error may occur.
    - Example) "my_image@MY IMAGE"
  - `-t`: "[Target file path]"
    - Target file path and file name
    - Example) "/folder/path/my_result.pdf"

- Requirements
  - [imagemagick](https://imagemagick.org/)

- Usage
  - `merge_pdf.sh` -f "[Folder Path]@[Label on Image]" -t "[Target file path]"

- Example
  
  ```shell
  ./merge_pdf.sh \
  -f "[input] current image@[input] current image" \
  -f "[input] p1 label@[input] prev label" \
  -f "[output] _predict__model_ref_local_tracking_model_022__run_leetaekyu_20210127_091102@[predict]" \
  -f "[target] current label@[target] current label" \
  -t "predict__model_ref_local_tracking_model_022__run_leetaekyu_20210127_091102.pdf"
  ```
