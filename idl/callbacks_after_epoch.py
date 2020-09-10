from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class EarlyStoppingAfter(EarlyStopping):
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
