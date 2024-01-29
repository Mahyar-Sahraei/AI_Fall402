import pickle as pkl
import numpy as np
from pandas import DataFrame
import hazm

class NLP:
    def __init__(self):
        self.pca = pkl.load(open("NLP/pca.pkl", "rb"))
        self.features = pkl.load(open("NLP/features.pkl", "rb"))
        self.stop_words = pkl.load(open("NLP/stop_words.pkl", "rb"))

        self.weights = np.load("NLP/model_weights.npy")
        self.bias = np.load("NLP/model_bias.npy")[0]

        self.normalized = []
        self.dtm = []

    def vectorize(self, tokens):
        vector = np.zeros(shape=len(self.features), dtype=np.int8)
        for i, word in enumerate(self.features):
            if word in tokens:
                vector[i] = min(tokens[word], 127)
        return vector

    def preprocess(self, df: DataFrame):
        # Normalization
        normalizer = hazm.Normalizer()
        for text in df["Text"]:
            text = normalizer.remove_specials_chars(text)
            text = normalizer.remove_diacritics(text)
            text = normalizer.decrease_repeated_chars(text)
            text = normalizer.seperate_mi(text)
            self.normalized.append(normalizer.normalize(text))

        # Stemming and Tokenizing
        stemmer = hazm.Stemmer()
        tokenized_data = []
        for text in self.normalized:
            tokens = hazm.word_tokenize(text)
            occurance_dict = {}
            for token in tokens:
                token = stemmer.stem(token)
                if (len(token) == 0) or (token in r"...[]\\;:,،()\?!{}<>#$\*-_") or (token in self.stop_words):
                    continue
                if token[0].isdigit() or token[0] in "۱۲۳۴۵۶۷۸۹۰":
                    token = r"%d"
                if token not in occurance_dict:
                    occurance_dict[token] = 0
                occurance_dict[token] += 1
            tokenized_data.append(occurance_dict)

        for tokens in tokenized_data:
            self.dtm.append(self.vectorize(tokens))

        self.dtm = np.array(self.dtm)
        self.dtm = self.pca.transform(self.dtm)

    def sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)

    def f(self, x):
        return self.sigmoid(np.dot(x, self.weights) + self.bias)

    def predict(self):
        categories = []
        for x in self.dtm:
            estimate = self.f(x)
            if estimate < 0.5:
                categories.append("Sport")
            else:
                categories.append("Politics")
        return categories

def predict(df: DataFrame):
    nlp = NLP()
    nlp.preprocess(df)
    return nlp.predict()