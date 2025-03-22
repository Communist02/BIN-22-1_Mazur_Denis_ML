import numpy as np
from processing import Processing

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Инициализация нейронной сети.

        :param input_size: количество входных нейронов
        :param hidden_size: количество нейронов в скрытом слое
        :param output_size: количество выходных нейронов
        """
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.hidden_output = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        return self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)

    def backward(self, X: np.ndarray, y: np.ndarray, output):
        output_delta = (y - output) * self.sigmoid_derivative(output)
        hidden_delta = output_delta.dot(self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta)
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True)
        self.weights_input_hidden += X.T.dot(hidden_delta)
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {np.mean(np.square(y - output))}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


if __name__ == "__main__":
    pc = Processing()
    text = 'Звезды отражались в её глазах. Раньше, еще месяц назад их не было видно в черте города. Свет фонарей и смог заглушали их слабое мерцание. Все изменилось. И одним из немногих плюсов сложившейся ситуации было мерцание звезд в её глазах, и воздух, кажется стал чище. У всех у нас когда то была работа, и был дом. У некоторых были дети. У Лены была дочка. Она работала барменшой, а по вечерам подрабатывала в "клубе знакомств". Попросту говоря - была проституткой. Теперь ей уже не приходится ездить по незнакомым клиентам, каждый раз перед дверью квартиры креститься, и молиться, что бы все прошло как надо. Это тоже плюс. Но теперь у неё нет дочки. Она потерялась в первые дни, как только все это начиналось. Лена была "на вызове", когда исчезло электричество. Никто еще не знал, что это серьезно. Мобильная связь не работала, город погрузился во тьму за окнами однакомнатной квартиры, в которой возбужденный мужчина кончал в презерватив, а Лена считала секунды до очередного вызова. Она не могла как обычно принять душ, и вызвать такси, и после осознания этого, просто начала одеваться. Белье по привычке было сложено одной кучкой рядом с кроватью. Мужчина, имя которого она не захотела запоминать сказал ей спасибо и открыл дверь, что то проворчав напоследок на "долбанных электриков"... Лене было очень приятно выйти на свежий воздух, после пропахшей перегаром комнатушки. Она шла по темным улицам города, шла на "базу" пешком, и эта непроглядная тьма вокруг для неё сейчас была отражением внутреннего состояния, и поэтому она наслаждалась этой прогулкой. Она еще не знала, что электричество и водоснабжение уже не восстановят. Она не могла подумать, что через три часа её пятилетняя дочка, испугавшись темноты и одиночества, выйдет из квартиры, и пропадет навсегда. Она еще не знала, что её поиски будут бесполезны и опасны... Она просто шла по улице.'
    tokens = pc.tokenize(text)  # Токенизация
    tokens = pc.lematize(tokens)  # Лемматизация
    tokens = pc.delete_stop_words(tokens)  # Удаление стоп слов
    vectors = np.array(pc.vectorize(tokens))
    vectors_dict = pc.vectorize_dict(tokens)
    inv_map = {v: k for k, v in vectors_dict.items()}
    n = np.linalg.norm(vectors)
    norm1 = vectors / np.linalg.norm(vectors)
    orders = [
    'Звезды отражались', 
    'Воздух стал', 
    'Она раньше шла', 
    'Дочка испугалась', 
    'Город погрузился', 
    'Она не могла месяц', 
    'Электричество исчезло', 
    'Поиск видно'
]

    result = [
        'глазах', 
        'чище', 
        'темным', 
        'одиночества', 
        'тьму', 
        'связь', 
        'связь', 
        'опасен'
    ]

    X = []
    y = []

    for text in orders:
        tokens = pc.tokenize(text)  # Токенизация
        tokens = pc.lematize(tokens)  # Лемматизация
        tokens = pc.delete_stop_words(tokens)  # Удаление стоп слов
        vectorsX = []
        for word in tokens:
            vectorsX.append(vectors_dict[word] / n)
        X.append(vectorsX)

    for text in result:
        tokens = pc.tokenize(text)  # Токенизация
        tokens = pc.lematize(tokens)  # Лемматизация
        tokens = pc.delete_stop_words(tokens)  # Удаление стоп слов
        vectorsY = []
        for word in tokens:
            vectorsY.append(vectors_dict[word] / n)
        y.append(vectorsY)

    X = np.array(X)
    y = np.array(y)
    nn = NeuralNetwork(input_size=2, hidden_size=10, output_size=1)
    nn.train(X, y, epochs=10000)
    print("Predictions:")
    predict = nn.predict(X)

    for pred in predict:
        r = round(pred[0] * n)
        print(inv_map[r])
