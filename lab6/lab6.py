import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import streamlit as st
import plotly.graph_objects as go
from datasets import Dataset

# Генерация фейковых данных
def generate_fake_news_data(num_samples=10000):
    categories = ['fake', 'real']
    data = {
        'text': [f"This is a {random.choice(categories)} news article." for _ in range(num_samples)],
        'label': [0 if random.choice(categories) == 'fake' else 1 for _ in range(num_samples)]
    }
    return pd.DataFrame(data)

# Подготовка данных
data = generate_fake_news_data(5000)
X = data['text']
y = data['label']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF для преобразования текста
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test, y_pred_nb)

# SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Токенизация
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)

# Создание Dataset для transformers
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': list(y_train)
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': list(y_test)
})

# Аргументы тренировки
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    save_total_limit=1,
    logging_steps=10
)

# Тренировка
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Streamlit Dashboard
st.title("Fake News Detection: Model Comparison")
st.write("### Accuracy Comparison of Naive Bayes, SVM, and BERT")
st.write(f"**Naive Bayes Accuracy:** {nb_accuracy:.4f}")
st.write(f"**SVM Accuracy:** {svm_accuracy:.4f}")
st.write(f"**BERT Accuracy:** {svm_accuracy:.4f}")

# Визуализация результатов
fig = go.Figure()

# Naive Bayes Predictions
fig.add_trace(go.Scatter(
    x=list(range(len(y_test))),
    y=y_pred_nb,
    mode='markers',
    name='Naive Bayes',
    marker=dict(color='blue', opacity=0.6)
))

# SVM Predictions
fig.add_trace(go.Scatter(
    x=list(range(len(y_test))),
    y=y_pred_svm,
    mode='markers',
    name='SVM',
    marker=dict(color='red', opacity=0.6)
))
# BERT
fig.add_trace(go.Scatter(
    x=list(range(len(y_test))),
    y=y_pred_svm,
    mode='markers',
    name='BERT',
    marker=dict(color='green', opacity=0.6)
))

fig.update_layout(
    title="Comparison of Naive Bayes , SVM Predictions and BERT",
    xaxis_title="Sample Index",
    yaxis_title="Predicted Label",
    showlegend=True
)

st.plotly_chart(fig)
