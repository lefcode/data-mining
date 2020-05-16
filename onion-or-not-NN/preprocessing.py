import string

import nltk
import pandas as pd

df = pd.read_csv('data/onion-or-not.csv')
data = pd.DataFrame(df)
rawData = data['text']


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
