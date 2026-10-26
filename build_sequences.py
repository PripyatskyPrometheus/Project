# build_sequences.py
import pandas as pd
import numpy as np
import time
import os
import re

data_path = "D:/dataset/labels/"
output_path = "D:/dataset/"

MAX_LEN = 250

# внутренние домены организации.
INTERNAL_DOMAINS = {"dtaa.com"}

print("ПЕРВЫЙ ЭТАП: ЗАГРУЗКА РАЗМЕЧЕННЫХ ЛОГОВ")

start = time.time()

def download_type(number, type_name, data_path, cols):
    print(f"\n{number}. Загрузка {type_name}_labeled.csv")
    df = pd.read_csv(data_path + f"{type_name}_labeled.csv", usecols=cols)
    print(f"Количество строк: {len(df)}")
    return df

logon  = download_type(1, "logon", data_path, ['user','date','pc','activity','is_true_bad'])
device = download_type(2, "device", data_path, ['user','date','pc','activity','is_true_bad'])
email  = download_type(3, "email", data_path, ['user','date','to','cc','bcc','attachments','is_true_bad'])
file   = download_type(4, "file", data_path, ['user','date','filename','is_true_bad'])
http   = download_type(5, "http", data_path, ['user','date','url','is_true_bad'])

print(f"\nВремя загрузки: {time.time()-start} секунд")

print(f"Положительных: logon={logon['is_true_bad'].sum()}, device={device['is_true_bad'].sum()}, http={http['is_true_bad'].sum()}, file={file['is_true_bad'].sum()}, email={email['is_true_bad'].sum()}")

print("\nВТОРОЙ ЭТАП: ПРЕОБРАЗОВАНИЕ ДАТ")

start = time.time()

for name, df in [('logon', logon), ('device', device), ('email', email), ('file', file), ('http', http)]:
    df['parsed_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S', cache=True)
    df.drop('date', axis=1, inplace=True)
    df['day'] = df['parsed_date'].dt.date
    print(f"{name}: даты преобразованы")

print(f"Время: {time.time()-start} секунд")

print("\nТРЕТИЙ ЭТАП: КЛАССИФИКАЦИЯ ДЕЙСТВИЙ")

start = time.time()

# device
device['action'] = np.where(
    device['activity'].astype(str).str.strip().str.lower() == 'connect',
    'device_connect',
    'device_disconnect'
)

# http
url = http['url'].astype(str).str.lower()
http['action'] = np.select(
    [
        url.str.contains('wikileaks|leak|anonymous', na=False),
        url.str.contains('job|career|indeed|monster', na=False),
        url.str.contains('facebook|twitter|linkedin|instagram', na=False),
        url.str.contains('dropbox|drive.google|mega|cloud', na=False),
        url.str.contains('gmail|mail|outlook', na=False),
        url.str.contains('keylog|spy|hack', na=False),
    ],
    ['http_leak', 'http_job', 'http_social', 'http_cloud', 'http_email_service', 'http_hack'],
    default='http_other'
)

# file
fname = file['filename'].astype(str)
ext = fname.str.split('.').str[-1].str.lower()
ext = ext.where(fname.str.contains('.'), 'other')
file['action'] = np.select(
    [
        ext.isin(['doc', 'docx', 'xls', 'xlsx', 'pdf', 'ppt', 'pptx', 'txt', 'rtf', 'csv', 'sql']),
        ext.isin(['zip', 'rar', '7z', 'tar', 'gz']),
        ext.isin(['exe', 'msi', 'bat', 'cmd', 'ps1', 'sh']),
        ext.isin(['jpg', 'jpeg', 'png', 'gif', 'bmp']),
    ],
    ['file_sensitive', 'file_archive', 'file_executable', 'file_image'],
    default='file_other'
)

# email: разбор всех адресов из to/cc/bcc
EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})')

def extract_domains(value):
    """Возвращает множество доменов из строки с адресами, разделёнными ; или ,"""
    if pd.isna(value):
        return set()
    text = str(value)
    return set(m.group(1).lower() for m in EMAIL_RE.finditer(text))

def classify_email(row):
    """email_external, если хотя бы один адресат (to+cc+bcc) вне INTERNAL_DOMAINS. Иначе email_internal."""
    domains = set()
    for col in ('to', 'cc', 'bcc'):
        domains |= extract_domains(row.get(col))
    if not domains:
        # нет валидных адресов — считаем external, чтобы не терять событие
        return 'email_external'
    if domains - INTERNAL_DOMAINS:
        return 'email_external'
    return 'email_internal'

email['action'] = email.apply(classify_email, axis=1)

# диагностика по доменам
_ext = email[email['action'] == 'email_external']
_int = email[email['action'] == 'email_internal']
print(f"email_internal: {len(_int)}, email_external: {len(_ext)}")
_sample_domains = set()
for col in ('to', 'cc', 'bcc'):
    for v in email[col].dropna().head(2000):
        _sample_domains |= extract_domains(v)
print(f"Примеры доменов (первые 10): {sorted(_sample_domains)[:10]}")

print(f"Время: {time.time()-start} секунд")

print("\nЧЕТВЁРТЫЙ ЭТАП: СБОР ДЕЙСТВИЙ ПО (USER, DAY)")

start = time.time()

def to_unified(df):
    return df[['user', 'day', 'parsed_date', 'action', 'is_true_bad']].copy()

all_actions = pd.concat([
    to_unified(device),
    to_unified(http),
    to_unified(file),
    to_unified(email),
], ignore_index=True)

print(f"Действий после классификации: {len(all_actions)}")
print(f"Положительных: {all_actions['is_true_bad'].sum()}")

# email_attach
email['attachments_num'] = pd.to_numeric(email['attachments'], errors='coerce').fillna(0)
email_att = email[email['attachments_num'] > 0].copy()
email_att['action'] = 'email_attach'

# email_night
email_night = email[(email['parsed_date'].dt.hour <= 6) | (email['parsed_date'].dt.hour >= 23)].copy()
email_night['action'] = 'email_night'

all_actions = pd.concat([
    all_actions,
    email_att[['user', 'day', 'parsed_date', 'action', 'is_true_bad']],
    email_night[['user', 'day', 'parsed_date', 'action', 'is_true_bad']],
], ignore_index=True)

print(f"После email_attach и email_night: {len(all_actions)}")
print(f"Положительных: {all_actions['is_true_bad'].sum()}")

# logon/logoff как действия (разметка CERT на них не вешается)
logon_actions = logon[['user', 'day', 'parsed_date']].copy()
logon_actions['action'] = 'logon'
logon_actions['is_true_bad'] = 0

logoff_actions = logon[logon['activity'].str.contains('Logoff', na=False)][['user', 'day', 'parsed_date']].copy()
logoff_actions['action'] = 'logoff'
logoff_actions['is_true_bad'] = 0

all_actions = pd.concat([
    all_actions,
    logon_actions[['user', 'day', 'parsed_date', 'action', 'is_true_bad']],
    logoff_actions[['user', 'day', 'parsed_date', 'action', 'is_true_bad']],
], ignore_index=True)

print(f"После logon/logoff: {len(all_actions)}")
print(f"Положительных: {all_actions['is_true_bad'].sum()}")

all_actions = all_actions.sort_values(['user', 'day', 'parsed_date']).reset_index(drop=True)

print(f"Время: {time.time()-start} секунд")

print("\nПЯТЫЙ ЭТАП: ГРУППИРОВКА ПО (USER, DAY)")

start = time.time()

seq_df = all_actions.groupby(['user', 'day']).agg(
    sequence=('action', list),
    has_anomaly=('is_true_bad', 'max'),
).reset_index()

print(f"Всего (user, day): {len(seq_df)}")
print(f"Аномальных: {seq_df['has_anomaly'].sum()}")

# обрезка до MAX_LEN
seq_df['full_length'] = seq_df['sequence'].apply(len)
seq_df['truncated'] = seq_df['full_length'] > MAX_LEN

truncated_count = seq_df['truncated'].sum()
truncated_anom = seq_df[seq_df['truncated']]['has_anomaly'].sum()

print(f"Длиннее {MAX_LEN}: {truncated_count} ({truncated_count/len(seq_df)*100:.2f}%)")
print(f"Из них аномальных: {truncated_anom}")

seq_df['sequence_length'] = seq_df['full_length'].clip(upper=MAX_LEN)
seq_df['sequence'] = seq_df['sequence'].apply(lambda x: x[:MAX_LEN])

print(f"Средняя длина (полная): {seq_df['full_length'].mean():.1f}")
print(f"Средняя длина (обрезанная): {seq_df['sequence_length'].mean():.1f}")
print(f"Максимальная длина (полная): {seq_df['full_length'].max()}")
print(f"Время: {time.time()-start} секунд")

print("\nШЕСТОЙ ЭТАП: СОХРАНЕНИЕ")

seq_df = seq_df[['user', 'day', 'sequence', 'has_anomaly', 'sequence_length', 'full_length', 'truncated']]

seq_df.to_csv(output_path + "sequences.csv", index=False)
print(f"Сохранено в {output_path}sequences.csv")

print(f"\nИтоговые счётчики:")
print(f"Всего (user, day): {len(seq_df)}")
print(f"Аномальных: {seq_df['has_anomaly'].sum()}")
print(f"Средняя длина (полная): {seq_df['full_length'].mean():.1f}")
print(f"Средняя длина (обрезанная): {seq_df['sequence_length'].mean():.1f}")
print(f"Максимальная длина (полная): {seq_df['full_length'].max()}")