from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


def hello_world():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    model = Sequential([
        Dense(1, input_shape=[1])
    ])

    # Metrics = ['accuracy'] is NOT right, need to find out why
    model.compile(optimizer='sgd', loss='mse')

    model.fit(xs, ys, epochs=500)

    print(model.predict([10.0]))


if __name__ == "__main__":
    hello_world()
