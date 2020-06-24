import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from human_vs_horse import bar_custom
import wget
import os
from zipfile import ZipFile
from myCallback import MyCallback
import matplotlib.pyplot as plt

INCEPTIONV3_WEIGHTS_URL = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
INCEPTIONV3_WEIGHTS_PATH = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

DATASET_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
DATASET_ZIP_PATH = '/tmp/cats_and_dogs_filtered.zip'
TRAIN_DATASET_PATH = 'cats_and_dogs_filtered/train'
TEST_DATASET_PATH = 'cats_and_dogs_filtered/validation'


def get_pretrained_model():
    pretrain_model = InceptionV3(
        input_shape=(150, 150, 3),
        include_top=False,
        weights=None
    )

    if not os.path.exists(INCEPTIONV3_WEIGHTS_PATH):
        wget.download(INCEPTIONV3_WEIGHTS_URL, INCEPTIONV3_WEIGHTS_PATH, bar=bar_custom)

    pretrain_model.load_weights(INCEPTIONV3_WEIGHTS_PATH)

    for layer in pretrain_model.layers:
        layer.trainable = False

    last_layer = pretrain_model.get_layer('mixed7')
    last_output = last_layer.output

    return pretrain_model.input, last_output


def prepare_data():
    if not os.path.exists(DATASET_ZIP_PATH):
        wget.download(DATASET_URL, DATASET_ZIP_PATH, bar=bar_custom)

    with ZipFile(DATASET_ZIP_PATH, 'r') as zipObj:
        zipObj.extractall()

    print()


def create_model(input, last_output):
    x = Flatten()(last_output)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(.2)(x)

    x = Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=input, outputs=x)

    return model

def get_generator():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=.2,
        zoom_range=.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATASET_PATH,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    validation_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    validation_generator = validation_datagen.flow_from_directory(
        TEST_DATASET_PATH,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    return train_generator, validation_generator


def plot_metrics(history):
    train_loss = history.history['loss']
    validation_loss =history.history['val_loss']

    train_acc = history.history['accuracy']
    validation_acc = history.history['val_accuracy']

    fig, (ax0, ax1) = plt.subplots(2, constrained_layout=True)

    ax0.set_title("Accuracy")
    ax0.plot(train_acc, 'r', label='Training Accuracy')
    ax0.plot(validation_acc, 'b', label='Validation Accuracy')
    ax0.grid(True)
    ax0.legend(loc=0)

    ax1.set_title("Loss")
    ax1.plot(train_loss, 'r', label='Training Loss')
    ax1.plot(validation_loss, 'b', label='Validation Loss')
    ax1.grid(True)
    ax1.legend(loc=0)
    plt.show()


if __name__ == '__main__':
    model_input, last_output = get_pretrained_model()
    model = create_model(model_input, last_output)

    prepare_data()
    train_generator, validation_generator = get_generator()

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
        metrics=['accuracy'])

    callback = MyCallback(0.95)
    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=50,
        callbacks=[callback]
    )

    plot_metrics(history)
