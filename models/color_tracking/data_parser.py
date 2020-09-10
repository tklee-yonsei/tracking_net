def training_data_parser(original_folder: str, target_folder: str) -> None:
    """
    트레이닝, 검증, 테스트 데이터를 위해 원본 데이터를 가공해 타겟 폴더에 저장한다.

    Parameters
    ----------
    original_folder : str
        원본 데이터가 있는 폴더. 내부에 image, label 폴더가 존재한다.
    target_folder : str
        데이터가 저장될 타겟 폴더. 내부에 image, prev_image, prev_label, label 폴더가 존재한다.
    """
    pass


def predict_data_parser(original_folder: str, target_folder: str) -> None:
    """
    예측 데이터를 위해 원본 데이터를 가공해 타겟 폴더에 저장한다.

    Parameters
    ----------
    original_folder : str
        원본 데이터가 있는 폴더. 내부에 image, label 폴더가 존재한다.
    target_folder : str
        데이터가 저장될 타겟 폴더. 내부에 image, prev_image, prev_label, label 폴더가 존재한다.
    """
    pass

