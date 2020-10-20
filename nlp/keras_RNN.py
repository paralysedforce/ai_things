from __future__ import print_function
import os
import argparse

import tensorflow as tf
from tensorflow import keras
import numpy as np

letters = "abcdefghijklmnopqrstuvyxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?-\"':;[] \t\n"
embeddings = {letter: i for i, letter in enumerate(letters)}
vocab = np.array(letters)

BATCH_SIZE = 64
EPOCHS = 30


def generate_model(X, Y):
    model = keras.models.Sequential([
        keras.layers.LSTM(8)
        ])
    model.compile(optimizer='adam', 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    model.fit(X, Y, epochs=2)


def preprocess_text(text):
    vectorized_text = np.array([embeddings[letter] 
                               for letter in text if letter in letters])


    # Split into sequences
    seq_length = 100
    
    char_dataset = tf.data.Dataset.from_tensor_slices(vectorized_text)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    dataset = sequences.map(lambda chunk: (chunk[:-1], chunk[1:]))

    buffer_size = 10000
    dataset = dataset.shuffle(buffer_size).batch(BATCH_SIZE, drop_remainder=True)
    print(dataset)

    return dataset

def build_model(batch_size=BATCH_SIZE):
    vocab_size = len(letters)
    embedding_dim = 256
    rnn_units = 1024

    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim,
                               batch_input_shape=[batch_size, None]),
        keras.layers.GRU(rnn_units,
                         return_sequences=True,
                         stateful=True,
                         recurrent_initializer='glorot_uniform'),
        keras.layers.Dense(vocab_size)
        ])
    
    def loss(labels, logits):
        return keras.losses.sparse_categorical_crossentropy(labels, logits,
                from_logits=True)

    model.compile(optimizer='adam', loss=loss)
    return model


def train_model(model, dataset):
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'checkpoint_{epoch}')

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

    model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

def main():
    parser = argparse.ArgumentParser(description='Character-based RNN model')
    parser.add_argument('filename', help='training data filename')
    args = parser.parse_args()

    with open(args.filename) as f:
        text = f.read()


    print("Text preprocessing started...", end='')
    dataset = preprocess_text(text)
    print("Done")

    print("Generating model...", end='')
    model = build_model()
    print("Done")

    print("Training model...", end='') 
    train_model(model, dataset)
    print("Done")

    model.save('./model')


if __name__ == '__main__':
    main()
