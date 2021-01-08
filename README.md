# Tracking Network

A network that tracks each pixel or object in a sequential image, such as a video clip

## Requirements

* Python &ge; 3.7
* Toolz &ge; 0.10.0
* TensorFlow &ge; 2.1.0
* Keras &ge; 2.3.1
* numpy &ge; 1.17.4

## Examples

```python
python _run/sample/semantic_segmentation/training_with_generator.py
python _run/experiment_name.py
```

## With TPU

### Prepare

```shell
sudo apt-get install libsm6 libxrender1 libfontconfig1
sudo apt-get install graphviz
pip3 install -r requirements.txt
```

### Example

* Run TPU cluster before.

```shell
ctpu up --zone=us-central1-b \
--tf-version=2.3.1 \
--tpu-size=v3-8 \
--name=leetaekyu-1-trainer \
--preemptible \
--tpu-only
```

#### U-Net L4

```shell
python3 _run/training_unet_l4_tpu.py \
--tpu_name "leetaekyu-1-trainer" \
--run_id "leetaekyu_20210108_221742
```

#### Ref Local Tracking 003

* Training everything from scratch.

    ```shell
    python3 _run/training_ref_local_tracking_003_tpu.py \
    --tpu_name "leetaekyu-1-trainer" \
    --run_id "leetaekyu_20210108_221742" \
    ```

* Use pre-trained U-Net backbone with fine tuning.

    ```shell
    python3 _run/training_ref_local_tracking_003_tpu.py \
    --tpu_name "leetaekyu-1-trainer" \
    --run_id "leetaekyu_20210108_221742" \
    --pretrained_unet_path "gs://cell_dataset/save/weights/training__model_unet_l4__run_leetaekyu_20210108_221742.epoch_78-val_loss_0.179-val_accuracy_0.974"
    ```

* Use pre-trained U-Net backbone and freeze U-Net weights.

    ```shell
    python3 _run/training_ref_local_tracking_003_tpu.py \
    --tpu_name "leetaekyu-1-trainer" \
    --run_id "leetaekyu_20210109_012720" \
    --pretrained_unet_path "gs://cell_dataset/save/weights/training__model_unet_l4__run_leetaekyu_20210108_221742.epoch_78-val_loss_0.179-val_accuracy_0.974" \
    --freeze_unet_model
    ```

## With docker

### Generate docker image

* Change folder

    ```shell
    cd code/tracking_net
    ```

* Build docker

    ```shell
    docker build .
    ```

* Docker image

    ```shell
    docker images

    <none>                  <none>                            53bf6b5d0f6a        44 hours ago        5.2GB
    nvidia/cuda             11.0-cudnn8-runtime-ubuntu18.04   848be2582b0a        12 days ago         3.6GB
    nvidia/cuda             10.2-base                         038eb67e1704        2 weeks ago         107MB
    nvidia/cuda             latest                            752312fac010        3 weeks ago         4.69GB
    nvidia/cuda             10.0-base                         0f12aac8787e        3 weeks ago         109MB
    ```

### Run docker image as container

* Run docker image as bash

    At [This project] folder. (`$(pwd)`)

    ```shell
    docker run \
        --gpus all \
        -it \
        --rm \
        -u $(id -u):$(id -g) \
        -v /etc/localtime:/etc/localtime:ro \
        -v $(pwd):/tracking_net \
        -p 6006:6006 \
        --workdir="/tracking_net" \
        [image id]
    ```

* (Optional) Or run docker using on shell.

    ```shell
    docker run \
        --gpus all \
        -it \
        --rm \
        -u $(id -u):$(id -g) \
        -v /etc/localtime:/etc/localtime:ro \
        -v $(pwd):/tracking_net \
        -p 6006:6006 \
        --workdir="/tracking_net" \
        d59e4204feec \
        python _run/sample/color_tracking/training_with_generator.py
    ```

* Detach from docker container

    Ctrl+p, Ctrl+q

* Attach to docker container again

    Show running docker containers.

    ```shell
    $ docker ps
    CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
    4c25ce8443e6        d59e4204feec        "/bin/bash"         4 hours ago         Up 4 hours                              zen_mendeleev
    ```

    Attach to container 4c25ce8443e6(Container id).

    ```shell
    docker attach 4c25ce8443e6
    docker attach $(docker ps -aq)
    ```

### (Optional) Tensorboard

* Run tensorboard on docker container

    ```shell
    docker exec [container id] tensorboard --logdir ./save/tf_logs/ --host 0.0.0.0 &
    ```

    Real example

    ```shell
    docker exec 7f1840636c9d tensorboard --logdir ./save/tf_logs/ --host 0.0.0.0 &
    docker exec $(docker ps -aq) tensorboard --logdir ./save/tf_logs/ --host 0.0.0.0 &
    ```

* Using ssh -L

    ```shell
    ssh -L 127.0.0.1:16006:0.0.0.0:6006 username@server
    ```

  * As mac client, I use "SSH Tunnel Manager" app to connect server.
