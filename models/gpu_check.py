import tensorflow as tf


def check_first_gpu():
    # GPU Setting
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        # 텐서플로가 첫 번째 GPU만 사용하도록 제한
        try:
            tf.config.experimental.set_visible_devices(gpus[0], "GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
            print(e)
            print("GPU Setting Error")
