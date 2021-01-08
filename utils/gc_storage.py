import os

from google.cloud import storage
from tensorflow.python.lib.io import file_io


def create_gcs_folder(bucket_name: str, path: str):
    gcs_client = storage.Client()
    bucket = gcs_client.get_bucket(bucket_name)
    blob = bucket.blob(path)
    blob.upload_from_string(
        "", content_type="application/x-www-form-urlencoded;charset=UTF-8"
    )


def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: str):
    """
    Downloads a blob from the bucket.

    Parameters
    ----------
    bucket_name : str
        Your bucket name
    source_blob_name : str
        Storage object name
    destination_file_name : str
        local path to file
    
    Examples
    --------
    >>> download_blob("your-bucket-name", "storage-object-name", "local/path/to/file")
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def upload_blob(bucket_name: str, source_file_name: str, destination_blob_name: str):
    """
    Uploads a file to the bucket.

    Parameters
    ----------
    bucket_name : [type]
        Your bucket name
    source_file_name : [type]
        local path to file
    destination_blob_name : [type]
        Storage object name

    Examples
    --------
    >>> upload_blob("your-bucket-name", "local/path/to/file", "storage-object-name")
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


def get_file_from_google_bucket(
    file_path: str, save_to_file_path: str, skip_if_exist: bool = True
):
    if (not skip_if_exist) or (
        skip_if_exist and (not os.path.isfile(save_to_file_path))
    ):
        loaded_file = file_io.FileIO(file_path, mode="rb",)
        save_to_file = open(save_to_file_path, "wb")
        save_to_file.write(loaded_file.read())
        save_to_file.close()

    return save_to_file_path
