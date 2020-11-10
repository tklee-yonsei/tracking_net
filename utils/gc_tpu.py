import tensorflow as tf


def tpu_initialize(tpu_address: str):
    """
    Initializes TPU for TF 2.x training.

    Parameters
    ----------
    tpu_address : str
        bns address of master TPU worker.

    Returns
    -------
    TPUClusterResolver
        A TPUClusterResolver.
    """
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices("TPU"))
    return resolver

