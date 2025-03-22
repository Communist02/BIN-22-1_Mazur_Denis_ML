import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
import math

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')


class Processing:
    def tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)

    def lematize(self, tokens: list[str]) -> list[str]:
        morph3 = MorphAnalyzer()
        lemmatized_words = [morph3.parse(
            word)[0].normal_form for word in tokens]
        return lemmatized_words

    def stemming(self, tokens: list[str]) -> list[str]:
        stemmer = SnowballStemmer("russian")
        lemmatized_words = [stemmer.stem(word) for word in tokens]
        return lemmatized_words

    def vectorize(self, tokens: list[str]) -> list[int]:
        dict_vectors = {}
        result = []
        for word in tokens:
            if word in dict_vectors.keys():
                result.append(dict_vectors[word])
            else:
                dict_vectors[word] = len(dict_vectors)
                result.append(dict_vectors[word])
        return result
    
    def vectorize_dict(self, tokens: list[str]) -> list[int]:
        dict_vectors = {}
        result = []
        for word in tokens:
            if word not in dict_vectors.keys():
                dict_vectors[word] = len(dict_vectors)
        return dict_vectors

    def delete_stop_words(self, tokens: list[str]) -> list[int]:
        stop_words = set(stopwords.words('russian')).union(['.', ',', ':', '?', '!'])
        return [word for word in tokens if word not in stop_words]

    def bag_of_words(self, tokens: list[str]) -> dict[str]:
        dict_words = {}
        for word in tokens:
            dict_words[word] = dict_words.setdefault(word, 0) + 1
        return dict_words

    def tf(self, tokens: list[str]) -> dict[str]:
        dict_words = self.bag_of_words(tokens)
        for word in dict_words:
            dict_words[word] /= len(tokens)
        return dict_words

    def idf(self, texts: list[list[str]]) -> dict[str]:
        dict_words = {}
        big_text = []
        for text in texts:
            big_text += list(set(text))
        for word in set(big_text):
            dict_words[word] = math.log(len(texts) / big_text.count(word))
        return dict_words

    def tf_idf(self, texts: list[list[str]], indexText: int) -> dict[str]:
        tf = self.tf(texts[indexText])
        idf = self.idf(texts)
        dict_words = {}
        for word in tf:
            dict_words[word] = tf[word] * idf[word]
        return dict_words
