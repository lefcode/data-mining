import string

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers.core import Dense
from keras.models import Sequential
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


def identity_tokenizer(text):
    return text


def prepare_input():
    clean = []
    for i in range(len(rawData)):
        clean.append((preprocessData(rawData[i])))
    clean = [list(clean[i]) for i in range(len(clean))]

    tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words='english', lowercase=False)

    allData = tfidf.fit_transform(clean).toarray()
    return allData


model = Sequential()
model.add(Dense(15, input_shape=(20237,), activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# TODO optimize model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

x = prepare_input()
y = np.array(df['label']).transpose()

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25)

history = model.fit(X_train, Y_train, epochs=500,
                    batch_size=32, verbose=2, validation_data=(X_test, Y_test))

# Evaluate model
scores = model.evaluate(X_test, Y_test, verbose=0)
