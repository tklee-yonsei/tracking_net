# Ref Tracking

## Training

### `train_tpu.py`

This is a trainer on TPU for Ref Tracking model which has following inputs and outputs.

- Inputs
  - Main image
  - Reference image
  - Reference label (scale 1/8)
  - Reference label (scale 1/4)
  - Reference label (scale 1/2)
  - Reference label (scale 1)
- Outputs
  - Main label

Examples

```shell
python3 ref_local_tracking/run/train_tpu.py \
--model_name "ref_local_tracking_model_011" \
--run_id "leetaekyu_20210109_012720" \
--tpu_name "leetaekyu-1-trainer" \
--with_shared_unet
```

```shell
python3 ref_local_tracking/run/train_tpu.py \
--model_name "ref_local_tracking_model_011" \
--run_id "leetaekyu_20210109_012720" \
--tpu_name "leetaekyu-1-trainer" \
--pretrained_unet_path "gs://cell_dataset/save/weights/training__model_unet_l4__run_leetaekyu_20210108_221742.epoch_78-val_loss_0.179-val_accuracy_0.974" \
--plot_sample \
--with_shared_unet \
--freeze_unet_model
```

```shell
python3 ref_local_tracking/run/train_tpu.py \
--model_name "ref_local_tracking_model_011" \
--bin_size 30 \
--batch_size 8 \
--training_epochs 200 \
--val_freq 1 \
--run_id "leetaekyu_20210109_012720" \
--ctpu_zone "us-central1-b" \
--tpu_name "leetaekyu-1-trainer" \
--gs_bucket_name "gs://cell_dataset" \
--training_dataset_folder "tracking_training" \
--validation_dataset_folder "tracking_validation" \
--pretrained_unet_path "gs://cell_dataset/save/weights/training__model_unet_l4__run_leetaekyu_20210108_221742.epoch_78-val_loss_0.179-val_accuracy_0.974" \
--plot_sample \
--with_shared_unet \
--freeze_unet_model
```

### `train2_tpu.py`

This is a trainer on TPU for Ref Tracking model which has following inputs and outputs.

- Inputs
  - Main image
  - Reference image
  - Reference label (scale 1)
- Outputs
  - Main label

Examples

```shell
python ref_local_tracking/run/training_tpu.py \
--model_name "" \
--tpu_name "leetaekyu-1-trainer" \
--run_id "leetaekyu_20210109_012720" \
--pretrained_unet_path "gs://cell_dataset/save/weights/training__model_unet_l4__run_leetaekyu_20210108_221742.epoch_78-val_loss_0.179-val_accuracy_0.974" \
--freeze_unet_model
```
