from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class EarlyStoppingAfter(EarlyStopping):
    """
    [summary]

    Parameters
    ----------
    EarlyStopping : [type]
        [description]
    
    Examples
    --------
    >>> apply_callbacks_after: int = 0
    >>> training_num_of_epochs: int = 200
    >>> val_freq: int = 1
    >>> early_stopping_patience: int = training_num_of_epochs // (10 * val_freq)
    >>> early_stopping: Callback = EarlyStoppingAfter(
    ...     patience=early_stopping_patience, verbose=1, after_epoch=apply_callbacks_after,
    ... )
    """

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        after_epoch=100,
    ):
        super(EarlyStoppingAfter, self).__init__(
            monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights
        )
        self.after_epoch = after_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.after_epoch:
            super().on_epoch_end(epoch, logs)


class ModelCheckpointAfter(ModelCheckpoint):
    """
    [summary]

    Parameters
    ----------
    ModelCheckpoint : [type]
        [description]

    Examples
    --------
    >>> import time
    >>> apply_callbacks_after: int = 0
    >>> model_name: str = "unet_l4"
    >>> run_id: str = time.strftime("%Y%m%d-%H%M%S")
    >>> training_id: str = "_training__model_{}__run_{}".format(model_name, run_id)
    >>> save_weights_folder: str = os.path.join("data", "weights")
    >>> model_checkpoint: Callback = ModelCheckpointAfter(
    ...     os.path.join(
    ...         save_weights_folder,
    ...         training_id[1:]
    ...         + ".epoch_{epoch:02d}-val_loss_{val_loss:.3f}-val_mean_iou_{val_mean_iou:.3f}.hdf5",
    ...     ),
    ...     verbose=1,
    ...     # save_best_only=True,
    ...     after_epoch=apply_callbacks_after,
    ... )
    >>> callback_list: List[Callback] = [model_checkpoint]
    """

    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        period=1,
        after_epoch=100,
    ):
        super(ModelCheckpointAfter, self).__init__(
            filepath, monitor, verbose, save_best_only, save_weights_only, mode, period
        )
        self.after_epoch = after_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.after_epoch:
            super().on_epoch_end(epoch, logs)


class ReduceLROnPlateauAfter(ReduceLROnPlateau):
    """
    [summary]

    Parameters
    ----------
    ReduceLROnPlateau : [type]
        [description]

    Examples
    --------
    >>> apply_callbacks_after: int = 0
    >>> reduce_lr_patience: int = 10
    >>> reduce_lr_cooldown: int = 5
    >>> reduce_lr: Callback = ReduceLROnPlateauAfter(
    ...     patience=reduce_lr_patience,
    ...     cooldown=reduce_lr_cooldown,
    ...     verbose=1,
    ...     after_epoch=apply_callbacks_after,
    ...     )
    >>> callback_list: List[Callback] = [reduce_lr]
    """

    def __init__(
        self,
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        min_lr=0,
        after_epoch=100,
        **kwargs
    ):
        super(ReduceLROnPlateauAfter, self).__init__(
            monitor,
            factor,
            patience,
            verbose,
            mode,
            min_delta,
            cooldown,
            min_lr,
            **kwargs
        )
        self.after_epoch = after_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.after_epoch:
            super().on_epoch_end(epoch, logs)
