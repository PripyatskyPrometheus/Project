# data_encoding.py
import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.model_selection import train_test_split


BASE_DIR = "C:/Users/user/dataset/"  # поменять при переезде на D:/dataset/

print("КОДИРОВАНИЕ ТОКЕНОВ И ПОДГОТОВКА ДАННЫХ для обучения")

print("\n1. Загрузка последовательностей из sequences.csv")
df = pd.read_csv(BASE_DIR + "sequences.csv")
df['sequence'] = df['sequence'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

df = df.drop(columns=['session_id'], errors='ignore')

assert df['has_anomaly'].isna().sum() == 0, "has_anomaly содержит NaN!!!"
df['has_anomaly'] = df['has_anomaly'].astype(int)
print(f"has_anomaly type: {df['has_anomaly'].dtype}, NaN: 0")

print(f"Всего (user, day): {len(df)}")
print(f"Аномальных: {df['has_anomaly'].sum()} ({df['has_anomaly'].mean()*100:.2f}%)")

print(df.head())

print("\n2. Создание словаря токенов")

all_tokens = [token for seq in df['sequence'] for token in seq]
unique_tokens = sorted(set(all_tokens))

print(f"\nУникальные Токены: {unique_tokens}")
print(f"Количество: {len(unique_tokens)}")

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
    'http_other': 0.1,
    'email_internal': 0.5,
    'email_external': 1.0,
    'http_job': 2.0,
    'http_social': 0.5,
    'email_attach': 1.0,
    'logon': 1.0,
    'logoff': 1.0,
    'file_sensitive': 2.0,
    'http_hack': 3.0,
    'http_leak': 3.0,
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

# Массив весов для каждого ID
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

df['padded'] = df['encoded'].apply(lambda x: np.array(pad_sequence(x), dtype=np.int32))

def mask_for_pad(seq, max_len=MAX_LEN):
    real = min(len(seq), max_len)
    return [1] * real + [0] * (max_len - real)

df['mask'] = df['encoded'].apply(lambda x: mask_for_pad(x))

truncated = (df['encoded_length'] > MAX_LEN).sum()
truncated_anom = df[df['encoded_length'] > MAX_LEN]['has_anomaly'].sum()
print(f"\nДней длиннее {MAX_LEN}: {truncated} ({truncated/len(df)*100:.2f}%)")
print(f"Из них аномальных: {truncated_anom}")
print(f"Аномальных последовательностей после padding: {df['has_anomaly'].sum()}")
print(f"Максимальная длина: {df['encoded_length'].max()}")

sample = df.iloc[0]['padded']
print(f"Пример: {sample[:10]}")
print(f"Длина после padding: {len(sample)}")

print("\n6. Разделение на train/val/test по пользователям")

user_anom = df.groupby('user')['has_anomaly'].max()
bad_users = user_anom[user_anom == 1].index.tolist()
good_users = user_anom[user_anom == 0].index.tolist()

print(f"Пользователей с аномалиями: {len(bad_users)}")
print(f"Пользователей без аномалий: {len(good_users)}")

bad_train, bad_test = train_test_split(bad_users, test_size=0.3, random_state=42)
bad_train, bad_val = train_test_split(bad_train, test_size=0.2, random_state=42)

good_train, good_test = train_test_split(good_users, test_size=0.3, random_state=42)
good_train, good_val = train_test_split(good_train, test_size=0.2, random_state=42)

train_users = bad_train + good_train
val_users = bad_val + good_val
test_users = bad_test + good_test

print(f"\nTrain: {len(train_users)} пользователей ({len(bad_train)} аномальных)")
print(f"Val: {len(val_users)} пользователей ({len(bad_val)} плохих)")
print(f"Test: {len(test_users)} пользователей ({len(bad_test)} плохих)")

user_anom.to_pickle(BASE_DIR + "user_anomaly.pkl")
print(f"\nРаспределение пользователей сохранено: {BASE_DIR}user_anomaly.pkl")

train_df = df[df['user'].isin(train_users)].copy()
val_df = df[df['user'].isin(val_users)].copy()
test_df = df[df['user'].isin(test_users)].copy()

n_neg = (train_df['has_anomaly'] == 0).sum()
n_pos = (train_df['has_anomaly'] == 1).sum()
pos_weight = n_neg / n_pos
print(f"\nМножитель штрафа за пропуск аномалий (pos_weight): {pos_weight:.1f}")
np.save(BASE_DIR + "pos_weight.npy", np.array([pos_weight]))
print(f"Сохранено: {BASE_DIR}pos_weight.npy")

user_anomalies = df.groupby('user')['has_anomaly'].sum()
print(f"\nПользователей с аномалиями: {(user_anomalies > 0).sum()} из {df['user'].nunique()}")
print(f"Максимум аномалий у одного пользователя: {user_anomalies.max()}")
print(f"Топ-5 пользователей по аномалиям:")
print(user_anomalies.sort_values(ascending=False).head(5))

print(f"\nTrain: {len(train_df)} (user, day) ({train_df['has_anomaly'].sum()} аномалий)")
print(f"Val: {len(val_df)} (user, day) ({val_df['has_anomaly'].sum()} аномалий)")
print(f"Test: {len(test_df)} (user, day) ({test_df['has_anomaly'].sum()} аномалий)")

print(f"\nПересечение train/val: {len(set(train_users) & set(val_users))}")
print(f"Пересечение train/test: {len(set(train_users) & set(test_users))}")
print(f"Пересечение val/test: {len(set(val_users) & set(test_users))}")

print("\n7. Полные признаки дня")

def add_full_length(df):
    df = df.sort_values(['user', 'day']).copy()
    if 'full_length' not in df.columns:
        df['full_length'] = df['sequence_length']
    return df

train_df = add_full_length(train_df)
val_df = add_full_length(val_df)
test_df = add_full_length(test_df)

max_len = float(train_df['full_length'].max())
for d in [train_df, val_df, test_df]:
    d['full_length_norm'] = d['full_length'] / max_len

# print("\n7b. Personal-baseline фичи (rolling 30d, shift(1), без меток)")

# извлекаем из sequence колонки для фич
def extract_daily_aggregates(d):
    """Считает агрегаты дня из sequence"""
    d = d.copy()
    d['n_files'] = d['sequence'].apply(lambda s: sum(1 for t in s if t.startswith('file_')))
    d['n_http'] = d['sequence'].apply(lambda s: sum(1 for t in s if t.startswith('http_')))
    d['n_file_sensitive'] = d['sequence'].apply(lambda s: sum(1 for t in s if t == 'file_sensitive'))
    d['n_email'] = d['sequence'].apply(lambda s: sum(1 for t in s if t.startswith('email_')))
    d['n_email_external'] = d['sequence'].apply(lambda s: sum(1 for t in s if t == 'email_external'))
    d['n_email_night'] = d['sequence'].apply(lambda s: sum(1 for t in s if t == 'email_night'))
    d['n_device_connect'] = d['sequence'].apply(lambda s: sum(1 for t in s if t == 'device_connect'))
    d['off_hours_ratio'] = np.where(d['n_email'] > 0, d['n_email_night'] / d['n_email'], 0.0)
    d['email_ext_ratio'] = np.where(d['n_email'] > 0, d['n_email_external'] / d['n_email'], 0.0)
    return d

train_df = extract_daily_aggregates(train_df)
val_df = extract_daily_aggregates(val_df)
test_df = extract_daily_aggregates(test_df)

def add_personal_baseline(d, min_periods=5, window=30):
    """Personal-baseline фичи
    
    Правила:
    - ratio-фичи: fallback 1.0 при отсутствии истории; log1p + clip(1e3) для выбросов.
    - dev-фичи: fallback 0.0 при отсутствии истории (NaN); sign*log1p(abs) для сжатия.
    - off_hours_dev / email_ext_dev: разность, NaN -> 0.
    """
    d = d.sort_values(['user', 'day']).copy()

    def rolling_prev(col):
        return (d.groupby('user')[col]
                 .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()))

    def ratio_feature(num_col, mean_col_name):
        mean = rolling_prev(num_col)
        d[mean_col_name] = mean
        val = np.where(mean.isna(), 1.0, d[num_col] / mean)
        val = np.clip(val, 0, 1e3)
        return np.log1p(val)

    def dev_feature(num_col, mean_col_name):
        mean = rolling_prev(num_col)
        d[mean_col_name] = mean
        dev = np.where(mean.isna(), 0.0, d[num_col] - mean)
        return np.sign(dev) * np.log1p(np.abs(dev))

    d['files_dev'] = dev_feature('n_files', 'files_mean_30d')

    d['http_ratio'] = ratio_feature('n_http', 'http_mean_30d')

    d['off_hours_mean_30d'] = rolling_prev('off_hours_ratio')
    d['off_hours_dev'] = (d['off_hours_ratio'] - d['off_hours_mean_30d']).fillna(0.0)

    d['email_ext_mean_30d'] = rolling_prev('email_ext_ratio')
    d['email_ext_dev'] = (d['email_ext_ratio'] - d['email_ext_mean_30d']).fillna(0.0)

    d['file_sensitive_dev'] = dev_feature('n_file_sensitive', 'file_sensitive_mean_30d')

    d['distinct_pcs_dev'] = dev_feature('n_device_connect', 'distinct_pcs_mean_30d')

    return d

train_df = add_personal_baseline(train_df)
val_df = add_personal_baseline(val_df)
test_df = add_personal_baseline(test_df)

for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
    print(f"\n{name}:")
    for c in ['files_dev', 'http_ratio', 'off_hours_dev', 'email_ext_dev', 'file_sensitive_dev', 'distinct_pcs_dev']:
        print(f"  {c}: mean={d[c].mean():.3f}, std={d[c].std():.3f}")

print("\n8. Сохранение данных")

with open(BASE_DIR + "token_to_id.pkl", 'wb') as f:
    pickle.dump(token_to_id, f)

with open(BASE_DIR + "id_to_token.pkl", 'wb') as f:
    pickle.dump(id_to_token, f)

print(f"\nСохраняем веса в {BASE_DIR}token_weights.npy")
np.save(BASE_DIR + "token_weights.npy", weights_array)

print("\nСохраняем датасеты")
train_df.to_pickle(BASE_DIR + "train_sequences.pkl")
val_df.to_pickle(BASE_DIR + "val_sequences.pkl")
test_df.to_pickle(BASE_DIR + "test_sequences.pkl")

print(f"{BASE_DIR}token_to_id.pkl")
print(f"{BASE_DIR}id_to_token.pkl")
print(f"{BASE_DIR}token_weights.npy")
print(f"{BASE_DIR}train_sequences.pkl")
print(f"{BASE_DIR}val_sequences.pkl")
print(f"{BASE_DIR}test_sequences.pkl")

print("\nКОДИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО")