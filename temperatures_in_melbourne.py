import wget
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Bidirectional, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import SGD

from human_vs_horse import bar_custom

TEMPERATURES_DATA_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
TEMPERATURES_DATA = 'daily-min-temperatures.csv'


def get_data():
    if not os.path.exists(TEMPERATURES_DATA):
        wget.download(TEMPERATURES_DATA_URL, TEMPERATURES_DATA, bar=bar_custom)

    # temperatures = []
    # time = []

    df = pd.read_csv(TEMPERATURES_DATA)
    # for i, row in enumerate(df.values):
    #     temperatures.append(row[1])
    #     time.append(i)
    temperatures = df['Temp'].values
    time = np.arange(temperatures.shape[0], dtype='float32')
    print(temperatures.shape, time.shape)
    print(temperatures[:10], time[:10])

    # return np.array(temperatures), np.array(time)
    return temperatures, time


def windowed_dataset(series, window_size, batch_size, shuffle_buffer_size):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(lambda window: (window[:-1], window[-1:]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds


def model_predict(series, window_size, batch_size, model):
    # series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    return model.predict(ds)


def get_best_learningrate(model):
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10 ** (epoch / 20)
    )
    model.compile(loss=Huber(), optimizer=SGD(lr=1e-8), metrics=['mae'])
    history = model.fit(training_set, epochs=100, callbacks=[lr_scheduler])
    lrs = 1e-8 * 10 ** (np.arange(100) / 20)

    plt.semilogx(history.history['lr'], history.history['loss'])
    plt.show()


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


batch_size = 30
window_size = 15
shuffle_buffer_size = 3000
if __name__ == '__main__':
    temperatures, time = get_data()
    print(type(temperatures))

    split = 3000
    training_data = temperatures[:split]
    training_time = time[:split]
    v_data = temperatures[split:]
    v_time = time[split:]

    training_set = windowed_dataset(training_data, window_size, batch_size, shuffle_buffer_size)

    # for _, batch in zip(range(1), training_set):
    #     print(batch)

    model = Sequential([
        Conv1D(60, 5, padding='causal', activation='relu', input_shape=[None, 1]),
        Bidirectional(LSTM(60, return_sequences=True)),
        Bidirectional(LSTM(60)),
        Dense(30, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1),
        Lambda(lambda x: x * 400)
    ])

    model.compile(loss=Huber(), optimizer=SGD(lr=1e-5), metrics=['mae'])
    history = model.fit(training_set, epochs=1)

    predictions = model_predict(temperatures[..., np.newaxis], window_size, batch_size, model)
    # get index [split-window_size:-1][0] from predictions
    predictions = predictions[split - window_size: -1, 0]
    # v_data = np.expand_dims(v_data, axis=-1)
    print(v_data.shape, predictions.shape)

    plt.figure(figsize=(10, 6))
    plot_series(v_time, v_data)
    plot_series(v_time, predictions)
    plt.show()

    print(tf.keras.metrics.mean_absolute_error(v_data, predictions).numpy())
    # print(tf.keras.metrics.mean_absolute_error(list(v_data), list(np.ravel(predictions))).numpy())
