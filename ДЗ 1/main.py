import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from pymorphy3 import MorphAnalyzer
import string

nltk.download('punkt_tab')
nltk.download('wordnet')

text = 'NLTK предлагает удобные инструменты для множества задач NLP: токенизация, стемминг, лемматизация, морфологический и синтаксический анализ, а также анализ настроений. Библиотека идеально подходит как для начинающих, так и для опытных разработчиков, предоставляя интуитивно понятный интерфейс и обширную документацию.'
english_text = 'Video player with the function of improving the quality of the hand-drawn image using the high-performance scaling algorithm of Anime4K.'


def lematize(text: str) -> list[str]:
    morph3 = MorphAnalyzer()
    tokens = word_tokenize(text)
    lemmatized_words = [morph3.parse(word)[0].normal_form for word in tokens]
    return lemmatized_words


def stemming(text: str) -> list[str]:
    stemmer = SnowballStemmer("russian")
    tokens = word_tokenize(text)
    lemmatized_words = [stemmer.stem(word) for word in tokens]
    return lemmatized_words


def ascii_tokenizer(text: str) -> list[str]:
    return [char for char in text if char in string.printable]


def ascii_vectorizer(text: str) -> list[str]:
    return [ord(char) for char in text if char in string.printable]


def tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def vectorize(tokens: list[str]) -> list[int]:
    dict_vectors = {}
    result = []
    for word in tokens:
        if word in dict_vectors.keys():
            result.append(dict_vectors[word])
        else:
            dict_vectors[word] = len(dict_vectors)
            result.append(dict_vectors[word])
    return result


print('Лемматизация:')
print(lematize(text))

print('Стемминг:')
print(stemming(text))

print('Токенизация всех символов из ASCII:')
print(ascii_tokenizer(english_text))

print('Векторизация всех символов из ASCII:')
print(ascii_vectorizer(english_text))

print('Векторизация текста после лемматизации:')
print(vectorize(lematize(text)))

print('Векторизация текста после стемминга:')
print(vectorize(stemming(text)))
