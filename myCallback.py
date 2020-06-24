import tensorflow as tf


class MyCallback(tf.keras.callbacks.Callback):
    target_accuracy = 0.85

    def __init__(self, target_accuracy=0.85):
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > self.target_accuracy:
            print('\n Accuracy reaches to {0:.0%}, stop training!'.format(self.target_accuracy))
            self.model.stop_training = True
