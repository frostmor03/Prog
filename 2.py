import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB

# Пути к папкам, содержащим текстовые файлы
path_neg = 'neg/'  # Путь к папке с отрицательными текстами
path_pos = 'pos/'  # Путь к папке с положительными текстами

# Создаем две строки для отрицательных и положительных обзоров
s1 = ''
s2 = ''

texts = []  # Список текстов
labels = []  # Список меток

# Получаем список файлов в каждой из папок
files_neg = glob.glob(os.path.join(path_neg, '*.txt'))
files_pos = glob.glob(os.path.join(path_pos, '*.txt'))

# Читаем содержимое файлов и объединяем в строки
for file_path in files_neg:
    with open(file_path, 'r', encoding='utf-8') as file:
        tem = file.read()
        s1 += ' ' + tem
        texts.append(tem)  # Добавляем текст в список
        labels.append(0)  # Добавляем метку 0 (отрицательный)

for file_path in files_pos:
    with open(file_path, 'r', encoding='utf-8') as file:
        tem = file.read()
        s2 += ' ' + tem
        texts.append(tem)  # Добавляем текст в список
        labels.append(1)  # Добавляем метку 1 (положительный)

# Создаем список корпусов
list_of_all = [s1, s2]
# print(list_of_all)

# Создаем список уникальных слов
words = list(set((s1 + ' ' + s2).split()))
# print(len(words))



# Инициализируем матрицу
results = np.zeros((len(texts), len(words)))
# print(results)
words_dict = {word: index for index, word in enumerate(words)}
# print(words_dict)

for i, text in enumerate(texts):
    for word in text.split():
        if word in words_dict:
            results[i, words_dict[word]] = 1

# Преобразовываем метки в массив NumPy
y = np.array(labels)

# Разбиваем данные на обучающий и тестовый наборы
X_train, X_test, Y_train, Y_test = train_test_split(results, y, test_size=0.2, random_state=0)

# Создаем и обучаем модель Bernoulli Naive Bayes
clf = BernoulliNB()
clf.fit(X_train, Y_train)

# Прогнозируем метки для тестового набора
y_pred = clf.predict(X_test)

# Вычисляем точность
accuracy = accuracy_score(Y_test, y_pred)

print("Accuracy:", accuracy)
