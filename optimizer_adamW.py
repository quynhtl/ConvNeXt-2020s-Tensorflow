import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.callbacks import Callback

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 20, 30 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3

    if epoch >= 30:
        lr *= 1e-2
    elif epoch >= 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def wd_schedule(epoch):
    """Weight Decay Schedule
    Weight decay is scheduled to be reduced after 20, 30 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        wd (float32): weight decay
    """
    wd = 1e-4

    if epoch >= 30:
        wd *= 1e-2
    elif epoch >= 20:
        wd *= 1e-1
    print('Weight decay: ', wd)
    return wd
    
@keras_export('keras.callbacks.WeightDecayScheduler')
class WeightDecayScheduler(Callback):
    """Weight Decay Scheduler.

    Arguments:
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            weight decay as output (float).
        verbose: int. 0: quiet, 1: update messages.

    ```python
    # This function keeps the weight decay at 0.001 for the first ten epochs
    # and decreases it exponentially after that.
    def scheduler(epoch):
      if epoch < 10:
        return 0.001
      else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    callback = WeightDecayScheduler(scheduler)
    model.fit(data, labels, epochs=100, callbacks=[callback],
              validation_data=(val_data, val_labels))
    ```
    """

    def __init__(self, schedule, verbose=0):
        super(WeightDecayScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'weight_decay'):
            raise ValueError('Optimizer must have a "weight_decay" attribute.')
        try:  # new API
            weight_decay = float(K.get_value(self.model.optimizer.weight_decay))
            weight_decay = self.schedule(epoch, weight_decay)
        except TypeError:  # Support for old API for backward compatibility
            weight_decay = self.schedule(epoch)
        if not isinstance(weight_decay, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.weight_decay, weight_decay)
        if self.verbose > 0:
            print('\nEpoch %05d: WeightDecayScheduler reducing weight '
                  'decay to %s.' % (epoch + 1, weight_decay))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['weight_decay'] = K.get_value(self.model.optimizer.weight_decay)