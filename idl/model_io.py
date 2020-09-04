from typing import Optional

import keras


def save_model(model_name: str, model: keras.models.Model):
    """
    모델을 저장합니다.

    Parameters
    ----------
    model_name : str
        [description]
    model : keras.models.Model
        [description]

    Examples
    --------
    >>> model = ...   # Keras Model
    >>> save_model(model, "model_name.json")
    """
    model_json = model.to_json()
    with open(model_name, "w") as json_file:
        json_file.write(model_json)


def load_model(
    model_name: str, with_weights_path: Optional[str] = None
) -> keras.models.Model:
    """
    모델을 불러옵니다.

    Parameters
    ----------
    model_name : str
        [description]
    with_weights_path : Optional[str], optional
        [description], by default None

    Returns
    -------
    keras.models.Model
        [description]

    Examples
    --------
    >>> model = load_model("model_name.json")
    >>> model_with_weights = load_model("model_name.json", "weights/model_weight.hdf5")
    """
    from keras.models import model_from_json

    json_file = open(model_name, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    if with_weights_path:
        model.load_weights(with_weights_path)
    return model
