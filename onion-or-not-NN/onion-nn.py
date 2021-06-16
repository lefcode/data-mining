import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers.core import Dense
from tensorflow.keras.models import Sequential
'''
from keras import backend as K
from keras.layers.core import Dense
from keras.models import Sequential
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

csv = pd.read_csv('data/onion-or-not.csv')
df = pd.DataFrame(csv)
rawData = df['text']


# Tokenizer function. Returns tokenized data
def tokenizeContent(contentsRaw):
    tokenized = nltk.tokenize.word_tokenize(contentsRaw)
    return tokenized


# Removing unnecessary stopwords
def removeStopWordsFromTokenized(contentsTokenized):
    stop_word_set = set(nltk.corpus.stopwords.words("english"))
    filteredContents = [word for word in contentsTokenized if word not in stop_word_set]
    return filteredContents


# Extra text normalization
def performPorterStemmingOnContents(contentsTokenized):
    porterStemmer = nltk.stem.PorterStemmer()
    filteredContents = [porterStemmer.stem(word) for word in contentsTokenized]
    return filteredContents


# Removing punctuation/extra text normalization
def removePunctuationFromTokenized(contentsTokenized):
    excludePuncuation = set(string.punctuation)

    # manually add additional punctuation to remove
    doubleSingleQuote = '\'\''
    doubleDash = '--'
    doubleTick = '``'
    newLine = '/n'

    excludePuncuation.add(doubleSingleQuote)
    excludePuncuation.add(doubleDash)
    excludePuncuation.add(doubleTick)
    excludePuncuation.add(newLine)

    filteredContents = [word for word in contentsTokenized if word not in excludePuncuation]
    return filteredContents


# Make all terms lower letter format
def convertItemsToLower(contentsRaw):
    filteredContents = [term.lower() for term in contentsRaw]
    return filteredContents


# Process data without writing inspection file information to file
def preprocessData(rawContents):
    cleaned = tokenizeContent(rawContents)
    cleaned = removeStopWordsFromTokenized(cleaned)
    cleaned = performPorterStemmingOnContents(cleaned)
    cleaned = removePunctuationFromTokenized(cleaned)
    cleaned = convertItemsToLower(cleaned)
    return cleaned


# used for tdifd
def identity_tokenizer(text):
    return text


# prepare tf-idf matrix into array format
def prepare_input():
    clean = []
    for i in range(len(rawData)):
        clean.append((preprocessData(rawData[i])))
    clean = [list(clean[i]) for i in range(len(clean))]

    tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)

    allData = tfidf.fit_transform(clean).toarray()
    return allData


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_score(y_true, y_pred):
    precision1 = precision(y_true, y_pred)
    recall1 = recall(y_true, y_pred)
    return 2 * ((precision1 * recall1) / (precision1 + recall1 + K.epsilon()))


def main():
    model = Sequential()
    # input shape is the volabulary length of tfidf
    model.add(Dense(15, input_shape=(20441,), activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[precision, recall, f1_score])

    x = prepare_input()
    y = np.array(df['label']).transpose()

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25)

    history = model.fit(X_train, Y_train, epochs=60,
                        batch_size=64, verbose=2, validation_data=(X_test, Y_test))

    # Evaluate model
    scores = model.evaluate(X_test, Y_test, verbose=0)

    # summarize history for metrics
    plt.figure(1)
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(311)
    plt.plot(history.history['val_f1_score'], color='red')
    plt.title('F1 Score')
    plt.ylabel('f1score')
    plt.xlabel('epoch')

    plt.subplot(312)
    plt.plot(history.history['val_recall'], color='green')
    plt.title('Recall Score')
    plt.ylabel('recall')
    plt.xlabel('epoch')

    plt.subplot(313)
    plt.plot(history.history['val_precision'], color='blue')
    plt.title('Precision Score')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.show()

    plt.figure(2)
    plt.plot(history.history['val_f1_score'], color='red')
    plt.plot(history.history['val_recall'], color='green')
    plt.plot(history.history['val_precision'], color='blue')
    plt.ylabel('Metrics')
    plt.xlabel('epoch')
    plt.legend(['f1 score', 'recall score', 'precision score'], loc='lower left')
    plt.show()


if __name__ == "__main__":
    main()
