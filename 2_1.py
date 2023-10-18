from keras.datasets import boston_housing
from sklearn.linear_model import Ridge

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)

mean = train_data.mean(axis=0) # Вычисление среднего значения каждого признака
train_data -= mean # Вычитание среднего из тренировочных данных. Нормализации данных, при которой данные центрируются относительно нуля.
std = train_data.std(axis=0) # Вычисление стандартного отклонения каждого признака в тренировочных данных.
train_data /= std # Деление тренировочных данных на их стандартное отклонение.Среднее значение 0 и стандартное отклонение 1 

test_data -= mean # Вычитание среднего из тестовых данных.
test_data /= std # Деление тестовых данных на стандартное отклонение тренировочных данных.

model = Ridge()
model.fit(train_data, train_targets)

print(model.score(test_data, test_targets))