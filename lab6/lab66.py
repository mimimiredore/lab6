import pandas as pd
import random
from faker import Faker

# Инициализация генератора случайных данных
fake = Faker()

# Количество записей
num_records = 10000

# Генерация случайных данных
data = {
    'text': [fake.text(max_nb_chars=200) for _ in range(num_records)],  # случайные текстовые данные
    'label': [random.choice([0, 1]) for _ in range(num_records)]  # случайные метки: 0 или 1
}

# Создание DataFrame
df = pd.DataFrame(data)

# Запись данных в CSV файл
df.to_csv('fake_news.csv', index=False)

print("CSV файл с данными успешно создан.")
