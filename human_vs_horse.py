import sys
import os
from zipfile import ZipFile

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import wget
import numpy as np

IMG_SIZE = (300, 300)
TRAIN_ZIP_URL = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
VALIDATION_ZIP_URL = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip'


def bar_custom(current, total, width=80):
    current /= 1000. * 1000.
    total /= 1000. * 1000.
    progress_message = "Downloading %.2f%% [%.2f / %.2f] MB" % (current / total * 100., current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


TRAIN_ZIP = '/tmp/horse-or-human.zip'
VALIDATION_ZIP = '/tmp/validation-horse-or-human.zip'

TRAIN_PATH = 'horse-or-human'
VALIDATION_PATH = 'validation-horse-or-human'


def prepare_data():
    if not os.path.exists(TRAIN_ZIP):
        wget.download(TRAIN_ZIP_URL, TRAIN_ZIP, bar=bar_custom)

    if not os.path.exists(VALIDATION_ZIP):
        wget.download(VALIDATION_ZIP_URL, VALIDATION_ZIP, bar=bar_custom)

    with ZipFile(TRAIN_ZIP, 'r') as zipObj:
        zipObj.extractall(TRAIN_PATH)

    with ZipFile(VALIDATION_ZIP, 'r') as zipObj:
        zipObj.extractall(VALIDATION_PATH)


def get_data_info(show=False):
    train_horse_dir = os.path.join(TRAIN_PATH, 'horses')
    train_human_dir = os.path.join(TRAIN_PATH, 'humans')

    validation_horse_dir = os.path.join(VALIDATION_PATH, 'horses')
    validation_human_dir = os.path.join(VALIDATION_PATH, 'humans')

    train_horse_names = os.listdir(train_horse_dir)
    train_human_names = os.listdir(train_human_dir)

    validation_horse_names = os.listdir(validation_horse_dir)
    validation_human_names = os.listdir(validation_human_dir)

    print('length of horse dir is {} and names are {}'.format(len(train_horse_names), train_horse_names[:10]))
    print('length of humans dir is {} and names are {}'.format(len(train_human_names), train_human_names[:10]))
    print('length of horse dir is {} and names are {}'.format(len(validation_horse_names), validation_horse_names[:10]))
    print(
        'length of humans dir is {} and names are {}'.format(len(validation_human_names), validation_human_names[:10]))

    nrows = 4
    ncols = 4

    if show:
        figure = plt.gcf()
        figure.set_size_inches(ncols * 4, nrows * 4)

        next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[:8]]
        next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_names[:8]]

        for i, img_path in enumerate(next_horse_pix + next_human_pix):
            sp = plt.subplot(nrows, ncols, i + 1)
            # sp.axis('off')
            img = mpimg.imread(img_path)
            plt.imshow(img)

        plt.show()


def get_generator(augmentation=False):
    if augmentation:
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
    else:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255
        )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        VALIDATION_PATH,
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode='binary'
    )

    return train_generator, test_generator


def train_model(epochs=8):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=[300, 300, 3]),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(1024, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    train_generator, test_generator = get_generator(augmentation=False)

    model.fit(
        train_generator,
        steps_per_epoch=65,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=16)

    return model


def plot_convolutions(model):
    train_horse_dir = os.path.join(TRAIN_PATH, 'horses')
    train_human_dir = os.path.join(TRAIN_PATH, 'humans')
    next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in os.listdir(train_horse_dir)]
    next_human_pix = [os.path.join(train_human_dir, fname) for fname in os.listdir(train_human_dir)]

    img_path = np.random.choice(next_horse_pix + next_human_pix)
    img = load_img(img_path, target_size=IMG_SIZE)
    x = np.expand_dims(img_to_array(img), axis=0)
    print(x.shape)

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    layer_names = [layer.name for layer in model.layers]
    features_map = activation_model.predict(x)
    f, axarr = plt.subplots(len(layer_names) - 3, sharex=False, sharey=False)
    # f.set_figwidth(20.)
    for i, (name, features) in enumerate(zip(layer_names, features_map)):
        size = features.shape[1]
        channel_size = features.shape[-1]

        if len(features.shape) == 4:
            display_grid = np.zeros((size, size * channel_size))
            for c in range(channel_size):
                conv = features[0, :, :, c]
                conv -= conv.mean()
                conv /= conv.std() + 1e-8
                conv *= 64
                conv += 128
                conv = np.clip(conv, 0, 255).astype('uint8')

                display_grid[:, c * size:c * size + size] = conv

            scale = 20. / channel_size
            # axarr[i].figure(figsize=(channel_size*scale, scale))
            # axarr[i].set_figheight(scale)
            # axarr[i].set_figwidth(channel_size*scale)
            axarr[i].set_title(name)
            axarr[i].imshow(display_grid, cmap='viridis')

    # plt.tight_layout(h_pad=8)
    plt.show()


if __name__ == '__main__':
    prepare_data()
    get_data_info()
    model = train_model(epochs=10)
    plot_convolutions(model)
