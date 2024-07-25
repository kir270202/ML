import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math

# Загрузка данных с использованием Pandas
def load_data(filename):
    df = pd.read_csv(filename, names=['label', 'data'], header=None, sep='\t')
    labels = df['label'].values.tolist()
    data = df['data'].values.tolist()
    return list(zip(data, labels))

# Преобразование текстовых данных в признаки с использованием nltk
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    features = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return set(features)

# Разбиение текстовых данных на слова и преобразование в мешок слов
def create_word_features(text):
    features = {}
    for word in preprocess_text(text):
        features[word] = True
    return features

# Обучение наивного байесовского классификатора
def train_naive_bayes_classifier(train_data):
    positive_features = []
    negative_features = []

    for text, label in train_data:
        features = create_word_features(text)
        if label == 'spam':
            positive_features.append((features, label))
        else:
            negative_features.append((features, label))

    # Вычисление вероятностей
    prior_positive = len(positive_features) / len(train_data)
    prior_negative = len(negative_features) / len(train_data)

    positive_word_probs = calculate_word_probs(positive_features)
    negative_word_probs = calculate_word_probs(negative_features)

    return prior_positive, prior_negative, positive_word_probs, negative_word_probs

# Вычисление вероятностей появления слова в классе
def calculate_word_probs(features_list):
    word_probs = {}
    total_documents = len(features_list)
    total_words = 0

    for features, _ in features_list:
        total_words += len(features)

        for word in features:
            if word in word_probs:
                word_probs[word] += 1
            else:
                word_probs[word] = 1

    for word in word_probs:
        word_probs[word] = (word_probs[word] + 1) / (total_words + total_documents)

    return word_probs

# Предсказание для наивного байесовского классификатора
def naive_bayes_predict(text, prior_positive, prior_negative, positive_word_probs, negative_word_probs):
    features = create_word_features(text)
    positive_prob = calculate_class_probability(features, prior_positive, positive_word_probs)
    negative_prob = calculate_class_probability(features, prior_negative, negative_word_probs)

    return 'spam' if positive_prob > negative_prob else 'ham'

# Вычисление вероятности принадлежности к классу
def calculate_class_probability(features, prior, word_probs):
    log_prob = math.log(prior)

    for word in features:
        if word in word_probs:
            log_prob += math.log(word_probs[word])
        else:
            # Лапласово сглаживание
            log_prob += math.log(1 / (len(word_probs) + len(features)))

    return log_prob

# Загрузка данных с использованием Pandas
filename = 'SMSSpamCollection.txt'
data = load_data(filename)

# Разбиение данных на обучающую и тестовую выборки
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Обучение наивного байесовского классификатора
prior_positive, prior_negative, positive_word_probs, negative_word_probs = train_naive_bayes_classifier(train_data)

# Тестирование
predictions = [naive_bayes_predict(text, prior_positive, prior_negative, positive_word_probs, negative_word_probs) for text, _ in test_data]

# Расчет точности
accuracy = accuracy_score([label for _, label in test_data], predictions)
print(f'Accuracy: {accuracy}')
accuracy_percent = accuracy * 100
print(f'Accuracy(%): {accuracy_percent:.2f}%')
