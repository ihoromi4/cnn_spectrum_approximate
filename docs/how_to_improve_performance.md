# How to improve performance

## 1. Улучшение через данные

1. Использовать больше данных
2. Сгенерировать больше данных
3. Сгенерировать дополнительные признаки
4. Нормализация данных

### 1.1 Больше данных

Алгоритмы глубокого обучения показывают лучшие результаты с ростом количества обучающих данных.

Часто при требовании получить высокую точность большую роль играют данные чем алгоритмы.

### 1.2 Генерация данных

Алгоритмы глубокого обучения достигают более высокой точности при большем количестве данных. Если нет возможности использовать больше данных, можно сгенерировать дополнительные данные искусственно.

Генерация подразумевает внесение изменений в исходные данные, например случайных отклонений. Этот подход называется `Data Augmentation`.

Это так же помогает избежать переобучения модели.

### 1.3 Дополнительные признаки

### 1.4 Нормализация данных

Данные нужно привести в диапазон значений соответствующий функции активации.

`sigmoid`: (0, 1)
`tanh`: (-1, 1)
`relu`: (0, inf)
