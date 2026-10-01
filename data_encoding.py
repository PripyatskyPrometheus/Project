# data_encoding.py
import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.model_selection import train_test_split


print("КОДИРОВАНИЕ ТОКЕНОВ И ПОДГОТОВКА ДАННЫХ для обучения")

print("\n1. Загрузка последовательностей из sequences.csv")
df = pd.read_csv("D:/dataset/sequences.csv")
df['sequence'] = df['sequence'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
print(f"Всего сессий: {len(df)}")
print(f"Аномальных: {df['has_anomaly'].sum()} ({df['has_anomaly'].mean()*100:.2f}%)")

print("\n2. Создание словаря токенов")

all_tokens = [token for seq in df['sequence'] for token in seq]
unique_tokens = sorted(set(all_tokens))
print(f"Уникальных токенов: {len(unique_tokens)}")
print(f"Токены: {unique_tokens}")

token_to_id = {token: i for i, token in enumerate(unique_tokens)}
id_to_token = {i: token for token, i in token_to_id.items()}

PAD_ID = len(token_to_id)
UNK_ID = len(token_to_id) + 1
token_to_id['PAD'] = PAD_ID
token_to_id['UNK'] = UNK_ID
id_to_token[PAD_ID] = 'PAD'
id_to_token[UNK_ID] = 'UNK'

print(f"PAD_ID: {PAD_ID}, UNK_ID: {UNK_ID}")
print(f"Итого размер словаря: {len(token_to_id)}")

print("\n3. Создание весов токенов")

token_weights = {
    'http_other': 0.1,        # пониженный
    'email_internal': 0.5,
    'email_external': 1.0,
    'http_job': 2.0,          # повышенный
    'http_social': 0.5,
    'email_attach': 1.0,
    'logon': 1.0,
    'logoff': 1.0,
    'file_sensitive': 2.0,    # повышенный
    'http_hack': 3.0,         # сильно повышенный
    'http_leak': 3.0,         # сильно повышенный
    'http_cloud': 1.5,
    'http_email_service': 1.0,
    'device_connect': 2.0,
    'device_disconnect': 2.0,
    'file_archive': 1.5,
    'file_executable': 2.0,
    'file_image': 0.5,
    'file_other': 0.5,
    'email_night': 1.5,
    'PAD': 0.0,
    'UNK': 0.0
}

# Создаём массив весов для каждого ID
weights_array = np.ones(len(token_to_id))
for token, weight in token_weights.items():
    if token in token_to_id:
        weights_array[token_to_id[token]] = weight

print(f"\nВеса созданы для {len(token_weights)} токенов")
print(f"http_other: {weights_array[token_to_id['http_other']]:.1f}")
print(f"http_hack: {weights_array[token_to_id['http_hack']]:.1f}")
print(f"file_sensitive: {weights_array[token_to_id['file_sensitive']]:.1f}")

print("\n4. Кодирование последовательностей")

df['encoded'] = df['sequence'].apply(lambda x: [token_to_id.get(t, UNK_ID) for t in x])
df['encoded_length'] = df['encoded'].apply(len)

print(f"\nСредняя длина: {df['encoded_length'].mean():.1f}")
print(f"Максимальная длина: {df['encoded_length'].max()}")

print("\n5. Обрезка/дополнение до 250 токенов")

MAX_LEN = 250

def pad_sequence(seq, max_len=MAX_LEN):
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [PAD_ID] * (max_len - len(seq))

df['padded'] = df['encoded'].apply(lambda x: pad_sequence(x))

sample = df.iloc[0]['padded']
print(f"Пример: {sample[:10]}...")
print(f"Длина после padding: {len(sample)}")

print("\n6. Разделение на train/val/test по пользователям")

unique_users = df['user'].unique()
train_users, test_users = train_test_split(unique_users, test_size=0.3, random_state=42)
train_users, val_users = train_test_split(train_users, test_size=0.2, random_state=42)

train_df = df[df['user'].isin(train_users)]
val_df = df[df['user'].isin(val_users)]
test_df = df[df['user'].isin(test_users)]

print(f"\nTrain: {len(train_df)} сессий ({train_df['has_anomaly'].sum()} аномалий)")
print(f"Val: {len(val_df)} сессий ({val_df['has_anomaly'].sum()} аномалий)")
print(f"Test: {len(test_df)} сессий ({test_df['has_anomaly'].sum()} аномалий)")

print(f"\nПересечение train/val: {len(set(train_users) & set(val_users))}")
print(f"Пересечение train/test: {len(set(train_users) & set(test_users))}")
print(f"Пересечение val/test: {len(set(val_users) & set(test_users))}")

print("\n7. Сохранение данных")

with open("D:/dataset/token_to_id.pkl", 'wb') as f:
    pickle.dump(token_to_id, f)

with open("D:/dataset/id_to_token.pkl", 'wb') as f:
    pickle.dump(id_to_token, f)

print("\nСохраняем веса в D:/dataset/token_weights.npy")
np.save("D:/dataset/token_weights.npy", weights_array)

print("\nСохраняем датасеты")
train_df.to_pickle("D:/dataset/train_sequences.pkl")
val_df.to_pickle("D:/dataset/val_sequences.pkl")
test_df.to_pickle("D:/dataset/test_sequences.pkl")

print(f"D:/dataset/token_to_id.pkl")
print(f"D:/dataset/id_to_token.pkl")
print(f"D:/dataset/token_weights.npy")
print(f"D:/dataset/train_sequences.pkl")
print(f"D:/dataset/val_sequences.pkl")
print(f"D:/dataset/test_sequences.pkl")

print("\nКОДИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО")