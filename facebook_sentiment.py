import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Embedding, Dropout, MaxPooling1D, LSTM
import wget
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dog_vs_cat import plot_metrics
from human_vs_horse import bar_custom
import os
import pandas as pd
import random
import numpy as np

FACEBOOK_DATASET_URL = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv'
FACEBOOK_DATASET = 'facebook_dataset.csv'

EMBEDDING_WEIGHTS_URL = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt'
EMBEDDING_WEIGHTS = 'glove.6B.100d.txt'

embedding_dims = 100
max_len = 16
padding_type = 'post'
trunc_type = 'post'
test_portion = .1
training_size = 160000


def get_facebook_comments_with_numpy():
    if not os.path.exists(FACEBOOK_DATASET):
        wget.download(FACEBOOK_DATASET_URL, FACEBOOK_DATASET, bar=bar_custom)

    df = pd.read_csv(FACEBOOK_DATASET, header=None)

    list_items = df.values[:, [0, 5]]

    print(list_items[34])
    np.random.shuffle(list_items)
    print(list_items[34])

    labels = np.where(list_items[:, 0] > 0, 1, 0)
    sentences = list_items[:, 1]

    print(sentences[34], labels[34])
    return sentences[:training_size], labels[:training_size]


def get_facebook_comments():
    if not os.path.exists(FACEBOOK_DATASET):
        wget.download(FACEBOOK_DATASET_URL, FACEBOOK_DATASET, bar=bar_custom)

    df = pd.read_csv(FACEBOOK_DATASET, header=None)

    # list_items = df.values[:, [0, 5]]
    list_items = []
    for row in df.values:
        sentiment = row[0]
        if sentiment == 0:
            sentiment = 0
        else:
            sentiment = 1
        item = [sentiment, row[5]]
        list_items.append(item)

    print(list_items[34])
    random.shuffle(list_items)
    print(list_items[34])

    labels = []
    sentences = []

    for _, item in zip(range(training_size), list_items):
        labels.append(item[0])
        sentences.append(item[1])

    print(sentences[34], labels[34])
    return sentences, labels


def get_embedding_weights(vocab_size, word_index):
    if not os.path.exists(EMBEDDING_WEIGHTS):
        wget.download(EMBEDDING_WEIGHTS_URL, EMBEDDING_WEIGHTS, bar=bar_custom)

    embeddings_index = {}
    with open(EMBEDDING_WEIGHTS, 'r') as fileObj:
        for line in fileObj:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32');
            embeddings_index[word] = embedding

    embeddings_matrix = np.zeros((vocab_size + 1, embedding_dims))
    for word, index in word_index.items():
        embeddings_vector = embeddings_index.get(word)
        if embeddings_vector is not None:
            embeddings_matrix[index] = embeddings_vector

    print(len(embeddings_matrix))
    return embeddings_matrix


if __name__ == '__main__':
    sentences, labels = get_facebook_comments_with_numpy()
    print(len(sentences), len(labels))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)

    word_index = tokenizer.word_index
    vocab_size = len(word_index)
    print(vocab_size)

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

    weights = get_embedding_weights(vocab_size, word_index)

    model = Sequential([
        Embedding(vocab_size + 1, embedding_dims, input_length=max_len, weights=[weights], trainable=False),
        Dropout(.2),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(pool_size=4),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    split = int(test_portion * len(padded))
    labels = np.expand_dims(labels, axis=-1)
    print(padded.shape, labels.shape)

    testing_sentences = padded[:split]
    testing_labels = labels[:split]
    training_sentences = padded[split:]
    training_labels = labels[split:]

    history = model.fit(training_sentences, training_labels, epochs=20,
                        validation_data=(testing_sentences, testing_labels))
    #
    # plot_metrics(history)
