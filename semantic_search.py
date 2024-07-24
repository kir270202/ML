import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cosine

# Загрузка универсального кодировщика предложений
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Загрузка данных
data = pd.read_csv("analyst_ratings_processed.csv", nrows=10000)

# Извлечение текстовых данных
texts = data["title"].tolist()

# Кодирование предложений с использованием универсального кодировщика
embeddings = np.array(embed(texts))

# Кластерный анализ с использованием KMeans
num_clusters = 3  # Количество кластеров
clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='cosine', linkage='complete')
clustering.fit(embeddings)

# Добавление меток кластеров в данные
data["cluster_label"] = clustering.labels_

# Реализация knn-классификатора
def knn_classifier(k, query_embedding):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings, data["cluster_label"])
    return knn.predict([query_embedding])[0]

# Семантический поиск по всему набору данных
def semantic_search_all(query_text):
    query_embedding = embed([query_text])[0]
    distances = [cosine(query_embedding, emb) for emb in embeddings]
    closest_idx = np.argmin(distances)
    return data.iloc[closest_idx]

# Семантический поиск внутри кластера для различных значений k
def semantic_search_in_cluster(query_text, k):
    query_embedding = embed([query_text])[0]
    cluster_label = knn_classifier(k, query_embedding)
    cluster_data = data[data["cluster_label"] == cluster_label]
    cluster_embeddings = embeddings[data["cluster_label"] == cluster_label]
    distances = [cosine(query_embedding, emb) for emb in cluster_embeddings]
    closest_idx = np.argmin(distances)
    return cluster_data.iloc[closest_idx]

# Пример поискового запроса
query = "10 Stocks To Watch For June 27, 2019"

# Семантический поиск по всему набору данных
result_all = semantic_search_all(query)
print("Semantic Search (All Data):")
print(result_all)

# Семантический поиск внутри кластера для различных значений k
k_values = [3, 5, 7]  #Выбираем значения k
for k in k_values:
    result_in_cluster = semantic_search_in_cluster(query, k)
    print(f"Semantic Search (Cluster, k={k}):")
    print(result_in_cluster)