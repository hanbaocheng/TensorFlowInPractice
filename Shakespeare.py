import tensorflow as tf
import wget
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.regularizers import l2

from dog_vs_cat import plot_metrics
from human_vs_horse import bar_custom

SONNETS_DATA_URL = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt'
SONNETS_DATA = 'sonnets.txt'


def get_data():
    if not os.path.exists(SONNETS_DATA):
        wget.download(SONNETS_DATA_URL, SONNETS_DATA, bar=bar_custom)

    with open(SONNETS_DATA, 'r') as fileObj:
        corpus = fileObj.read().lower().split('\n')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []

    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x)for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, labels = input_sequences[:, :-1], input_sequences[:, -1:]
    labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    return tokenizer, predictors, labels, total_words, max_sequence_len

def generator(tokenizer, max_sequence_len,  model):
    seed_text = 'you are my sunshine, my only sunshine'
    next_words = 100

    for i in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        prediction = model.predict(token_list)
        prediction = np.argmax(prediction)

        next_word = ''
        for word, index in tokenizer.word_index.items():
            if index == prediction:
                next_word = word
                break
        seed_text = seed_text + ' ' + next_word

    print(seed_text)


if __name__ == '__main__':
    tokenizer, predictors, labels, total_words, max_sequence_len = get_data()

    print(labels.shape)
    model = Sequential([
        Embedding(total_words, 120, input_length=max_sequence_len - 1),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(.2),
        Bidirectional(LSTM(128)),
        Dense(total_words // 2, kernel_regularizer=l2(0.0003), activation='relu'),
        Dense(total_words, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(predictors, labels, epochs=100)

    plot_metrics(history)

    generator(tokenizer, max_sequence_len, model)


