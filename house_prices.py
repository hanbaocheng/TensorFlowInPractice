from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


def house_prices(size):
    model = Sequential([
        Dense(1, input_shape=[1])
    ])

    sizes = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
    prices = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4], dtype=float)

    model.compile(optimizer='sgd', loss='mse')

    model.fit(sizes, prices, epochs=500)

    print(model.predict([size]))


if __name__ == '__main__':
    house_prices(14)
