from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
import torch

# Загрузка данных
data = pd.read_csv('fake_news.csv')

# Разделение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Токенизация
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# Создание объектов Dataset
train_data = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
test_data = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})

# Применение токенизации
train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

# Удаление ненужных колонок (оставляем только input_ids, attention_mask, и labels)
train_data = train_data.remove_columns(['text'])
test_data = test_data.remove_columns(['text'])
train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Загрузка модели
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Настройки обучения
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",  # Используйте eval_strategy вместо evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
)


# Создание Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# Обучение модели
trainer.train()

# Оценка модели
metrics = trainer.evaluate()
print("Evaluation Metrics:", metrics)
