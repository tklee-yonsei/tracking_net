from tensorflow.keras.models import Model
from tensorflow.python.lib.io import file_io


class ModelGSSave(Model):
    def __init__(self, *args, **kwargs):
        super(ModelGSSave, self).__init__(*args, **kwargs)

    def save(
        self,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
    ):
        if str.startswith(filepath, "gs://"):
            super().save(
                filepath.split("/")[-1],
                overwrite,
                include_optimizer,
                save_format,
                signatures,
                options,
            )
            with file_io.FileIO(filepath.split("/")[-1], mode="rb") as input_f:
                with file_io.FileIO(filepath, mode="wb+") as output_f:
                    output_f.write(input_f.read())
        else:
            super().save(
                filepath, overwrite, include_optimizer, save_format, signatures, options
            )
