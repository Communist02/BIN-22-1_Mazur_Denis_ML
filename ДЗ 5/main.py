# The AG News Corpus is a popular dataset commonly used for text classification tasks in Natural Language Processing (NLP). It consists of news articles collected from the AG's corpus of news articles on the web, categorized into four classes: World, Sports, Business, and Science/Technology. Each article is accompanied by a title and a short description, making it suitable for tasks like topic classification and sentiment analysis. With its diverse range of topics and well-labeled categories, the AG News Corpus serves as a valuable resource for training and evaluating machine learning models in various NLP applications.

# Description:

# Dataset: AG News Corpus
# Source: AG's corpus of news articles on the web.
# Content: News articles categorized into World, Sports, Business, and Science/Technology.
# Labels: Four class labels representing different news categories.
# Scope: Covers a broad range of current events and topics.
# Size: Typically contains thousands of articles.
# Language: Primarily in English.

from processing import Processing
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('ДЗ 5/train.csv')
pc = Processing()

for index, row in df.iterrows():
    text = row['Title']
    tokens = pc.tokenize(text)  # Токенизация
    tokens = pc.lematize(tokens)  # Лемматизация
    tokens = pc.delete_stop_words(tokens)  # Удаление стоп слов

    row['Title'] = ' '.join(tokens)

    text = row['Description']
    tokens = pc.tokenize(text)  # Токенизация
    tokens = pc.lematize(tokens)  # Лемматизация
    tokens = pc.delete_stop_words(tokens)  # Удаление стоп слов

    row['Description'] = ' '.join(tokens)

y = df['Class Index']
x = df.copy().drop('Class Index', axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)
