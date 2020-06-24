from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from myCallback import MyCallback

desired_width = 320
# pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)


def plot_convolutions(model, images, images_num=3, layers_num=4, convolution=1):
    figure, axarr = plt.subplots(3, 4)

    output_layers = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=output_layers)

    for image in range(images_num):
        outputs = activation_model(np.expand_dims(images[image], axis=0))
        for layer in range(layers_num):
            f1 = outputs[layer]
            axarr[image, layer].imshow(f1[0, :, :, convolution], cmap='inferno')
            # axarr[image, layer].imshow(f1[0, :, :, convolution])
            axarr[image, layer].grid(False)

    plt.show()


def create_dnn():
    model = Sequential([
        Flatten(input_shape=[28, 28]),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])

    return model


def create_cnn():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=[28, 28, 1]),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])

    return model


def fashion_mnist():
    mnist = tf.keras.datasets.fashion_mnist
    # mnist = tf.keras.datasets.mnist
    callbacks = MyCallback()

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    plt.imshow(train_images[15])
    # plt.show()
    print(train_images[15])

    # model = dnn()

    # for CNN, add one more dimension
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    model = create_cnn()

    train_images = train_images / 255
    test_images = test_images / 255

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=8, validation_data=(test_images, test_labels), callbacks=[callbacks])

    model.evaluate(test_images, test_labels)
    classifications = np.argmax(model.predict(test_images), axis=-1)
    print(classifications[5], test_labels[5])

    plot_convolutions(model, test_images[:3], convolution=1)


if __name__ == '__main__':
    fashion_mnist()
