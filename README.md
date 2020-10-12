# Tracking Network

U-Net 기반의 일련의 연속적인 이미지에서 각 인스턴스 객체를 분리 및 추적하는 네트워크

## 요구사항

* Python &ge; 3.7
* Toolz &ge; 0.10.0
* TensorFlow &ge; 2.1.0
* Keras &ge; 2.3.1
* numpy &ge; 1.17.4

## 실행 예시

```python
python _run/sample/semantic_segmentation/training_with_generator.py
python _run/experiment_name.py
```

## With docker

### Generate docker image

```shell
cd code/tracking_net
docker build .
```

```shell
cd code/tracking_net
docker run \
    --gpus all \
    -it \
    --rm \
    -u $(id -u):$(id -g) \
    -v /etc/localtime:/etc/localtime:ro \
    -v $(pwd):/code/tracking_net \
    --workdir="/code/tracking_net" \
    [image id] \
    python _run/training_001.py
```
