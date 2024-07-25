import numpy as np

def sigmoid(x): # логистическая функция
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr #Learning rate
        self.n_iters = n_iters #Количество итераций
        self.weights = None #Весовые коэффициенты
        self.bias = None #Смещение

    def fit(self, X, y): #Обучение модели
        n_samples, n_features = X.shape #Количество строк и столбцов
        self.weights = np.zeros(n_features) #Инициализируем весовые коэффициенты нулями
        self.bias = 0 #Инициализируем смещение

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias #Предсказание по линейному закону
            predictions = sigmoid(linear_pred) #Предсказание по логистическому закону

            #Вычисляем градиент
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw #Изменяем значения весовых коэффициентов
            self.bias = self.bias - self.lr*db #Изменяем значения смещения


    def predict(self, X): #Получаем предсказание
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]  #Классификация исходя из графика сигмоиды
        return class_pred
