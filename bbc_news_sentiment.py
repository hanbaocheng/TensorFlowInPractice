import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from human_vs_horse import bar_custom
import wget
import os
import pandas as pd
import numpy as np
import random

BBC_TEXT_URL = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv'
BBC_TEXT = 'bbc-text.csv'
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

vocab_size = 10000
embedding_dim = 16
max_len = 120
padding_type = 'post'
trunc_type = 'post'
oov_tok = '<OOV>'
training_portion = .8


def remove_stopwords(sentence):
    for word in stopwords:
        sentence = sentence.replace(' ' + word + ' ', ' ')
        sentence = sentence.replace('   ', ' ')
    return sentence


def get_sentences():
    if not os.path.exists(BBC_TEXT):
        wget.download(BBC_TEXT_URL, BBC_TEXT, bar=bar_custom)

    df = pd.read_csv(BBC_TEXT)

    category = df['category']
    text = df['text'].apply(remove_stopwords)

    print(category.shape, text.shape)
    return category, text


def get_dataset(category, text):
    # print(text[1])
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(text)
    corpus = tokenizer.word_index
    print(len(corpus))

    sequences = tokenizer.texts_to_sequences(text)
    padded_sentences = pad_sequences(sequences, maxlen=max_len, truncating=trunc_type, padding=padding_type)
    print(padded_sentences.shape)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(category)
    label_sequences = label_tokenizer.texts_to_sequences(category)
    label_word_index = label_tokenizer.word_index

    print(label_word_index)
    print(len(padded_sentences), len(label_sequences))

    train_size = int(padded_sentences.shape[0] * training_portion)

    training_sentences = np.array(padded_sentences[:train_size])
    validation_sentences = np.array(padded_sentences[train_size:])

    training_labels = np.array(label_sequences[:train_size])
    validation_labels = np.array(label_sequences[train_size:])

    return training_sentences, training_labels, validation_sentences, validation_labels


if __name__ == '__main__':

    category, text = get_sentences()
    train_sentences, training_labels, validation_sentences, validation_labels = get_dataset(category, text)

    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(6, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_sentences, training_labels, epochs=10, validation_data=(validation_sentences, validation_labels))

