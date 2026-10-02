# build_sequences.py
import pandas as pd
import numpy as np
import time
import os

data_path = "D:/dataset/labels/"
output_path = "D:/dataset/"

print("ПЕРВЫЙ ЭТАП: ЗАГРУЗКА РАЗМЕЧЕННЫХ ЛОГОВ")

start = time.time()

def download_type(number, type_name, data_path, cols):
    print(f"\n{number}. Загрузка {type_name}_labeled.csv")
    df = pd.read_csv(data_path + f"{type_name}_labeled.csv", usecols=cols)
    print(f"Количество строк: {len(df)}")
    return df

logon  = download_type(1, "logon", data_path, ['user','date','pc','activity','is_true_bad'])
device = download_type(2, "device", data_path, ['user','date','pc','activity','is_true_bad'])
email  = download_type(3, "email", data_path, ['user','date','to','attachments','is_true_bad'])
file   = download_type(4, "file", data_path, ['user','date','filename','is_true_bad'])
http   = download_type(5, "http", data_path, ['user','date','url','is_true_bad'])

print(f"\nВремя загрузки: {time.time()-start} секунд")

print("\nВТОРОЙ ЭТАП: ПРЕОБРАЗОВАНИЕ ДАТ")

start = time.time()

for name, df in [('logon', logon), ('device', device), ('email', email), ('file', file), ('http', http)]:
    df['parsed_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S', cache=True)
    df.drop('date', axis=1, inplace=True)
    print(f"{name}: даты преобразованы")

print(f"Время: {time.time()-start} секунд")

print("\nТРЕТИЙ ЭТАП: ПОИСК СЕССИЙ")

start = time.time()

logon_clean = logon[logon['activity'].str.contains('Logon|Logoff', na=False)].copy()
logon_clean['activity'] = logon_clean['activity'].str.strip()

print(f"Строк с Logon и Logoff {len(logon_clean)} штук")

logon_clean = logon_clean.sort_values(['user', 'parsed_date'])

sessions = []
current_user = None
current_logon = None

for _, row in logon_clean.iterrows():
    user = row['user']
    activity = row['activity']
    dt = row['parsed_date']
    
    # Если новый пользователь — сбрасываем
    if user != current_user:
        if current_logon is not None:
            # Незакрытая сессия — закрываем по концу дня
            sessions.append({
                'user': current_user,
                'start': current_logon['parsed_date'],
                'end': current_logon['parsed_date'].replace(hour=23, minute=59, second=59),
                'pc': current_logon['pc'],
                'is_true_bad': current_logon['is_true_bad']
            })
        current_user = user
        current_logon = None
    
    if 'Logon' in activity:
        current_logon = row
    elif 'Logoff' in activity and current_logon is not None:
        sessions.append({
            'user': user,
            'start': current_logon['parsed_date'],
            'end': dt,
            'pc': current_logon['pc'],
            'is_true_bad': current_logon['is_true_bad']
        })
        current_logon = None

if current_logon is not None:
    sessions.append({
        'user': current_user,
        'start': current_logon['parsed_date'],
        'end': current_logon['parsed_date'].replace(hour=23, minute=59, second=59),
        'pc': current_logon['pc'],
        'is_true_bad': current_logon['is_true_bad'],
    })

sessions_df = pd.DataFrame(sessions)
print(f"Всего сессий: {len(sessions_df)}")
print(f"Уникальных пользователей: {sessions_df['user'].nunique()}")
print(f"Время: {time.time()-start} секунд")

print("\nЧЕТВЁРТЫЙ ЭТАП: СБОР ДЕЙСТВИЙ ВНУТРИ СЕССИЙ")

start = time.time()

sessions_df = sessions_df.reset_index(drop=True)
sessions_df['session_id'] = sessions_df.index

sessions_sorted = sessions_df.sort_values('start').reset_index(drop=True)

def attach_session(events, sessions):
    """Приклеивает session_id к каждому событию:
    событие попадает в сессию, если start <= event_time <= end и user совпадает."""
    ev = events.sort_values('parsed_date').reset_index(drop=True)
    ss = sessions[['user', 'start', 'end', 'session_id']].sort_values('start').reset_index(drop=True)
    
    merged = pd.merge_asof(
        ev,
        ss,
        left_on='parsed_date',
        right_on='start',
        by='user',
        direction='backward'
    )
    merged = merged[merged['parsed_date'] <= merged['end']]
    return merged

device_s = attach_session(device, sessions_sorted)
http_s   = attach_session(http, sessions_sorted)
file_s   = attach_session(file, sessions_sorted)
email_s  = attach_session(email, sessions_sorted)

print(f"device_s: {len(device_s)}, http_s: {len(http_s)}, file_s: {len(file_s)}, email_s: {len(email_s)}")

# device
device_s['action'] = np.where(
    device_s['activity'].astype(str).str.strip().str.lower() == 'connect',
    'device_connect',
    'device_disconnect'
)

# http
url = http_s['url'].astype(str).str.lower()
http_s['action'] = np.select(
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
fname = file_s['filename'].astype(str)
ext = fname.str.split('.').str[-1].str.lower()
ext = ext.where(fname.str.contains('.'), 'other')
file_s['action'] = np.select(
    [
        ext.isin(['doc', 'docx', 'xls', 'xlsx', 'pdf', 'ppt', 'pptx', 'txt', 'rtf', 'csv', 'sql']),
        ext.isin(['zip', 'rar', '7z', 'tar', 'gz']),
        ext.isin(['exe', 'msi', 'bat', 'cmd', 'ps1', 'sh']),
        ext.isin(['jpg', 'jpeg', 'png', 'gif', 'bmp']),
    ],
    ['file_sensitive', 'file_archive', 'file_executable', 'file_image'],
    default='file_other'
)

# email
to = email_s['to'].astype(str)
email_s['action'] = np.where(
    to.str.contains('dtaa.com', na=False),
    'email_internal',
    'email_external'
)

def to_unified(df):
    return df[['session_id', 'user', 'parsed_date', 'action', 'is_true_bad']].copy()

all_actions = pd.concat([
    to_unified(device_s),
    to_unified(http_s),
    to_unified(file_s),
    to_unified(email_s),
], ignore_index=True)

print(f"Всего действий: {len(all_actions)}")

# email_attach
email_s['attachments_num'] = pd.to_numeric(email_s['attachments'], errors='coerce').fillna(0)
email_att = email_s[email_s['attachments_num'] > 0].copy()
email_att['action'] = 'email_attach'

# email_night
email_night = email_s[(email_s['parsed_date'].dt.hour <= 6) | (email_s['parsed_date'].dt.hour >= 23)].copy()
email_night['action'] = 'email_night'

all_actions = pd.concat([
    all_actions,
    email_att[['session_id', 'user', 'parsed_date', 'action', 'is_true_bad']],
    email_night[['session_id', 'user', 'parsed_date', 'action', 'is_true_bad']],
], ignore_index=True)

logon_actions = sessions_df[['session_id', 'user', 'start']].copy()
logon_actions = logon_actions.rename(columns={'start': 'parsed_date'})
logon_actions['action'] = 'logon'
logon_actions['is_true_bad'] = 0

logoff_actions = sessions_df[['session_id', 'user', 'end']].copy()
logoff_actions = logoff_actions.rename(columns={'end': 'parsed_date'})
logoff_actions['action'] = 'logoff'
logoff_actions['is_true_bad'] = 0

all_actions = pd.concat([
    all_actions,
    logon_actions[['session_id', 'user', 'parsed_date', 'action', 'is_true_bad']],
    logoff_actions[['session_id', 'user', 'parsed_date', 'action', 'is_true_bad']],
], ignore_index=True)

all_actions = all_actions.sort_values(['session_id', 'parsed_date']).reset_index(drop=True)

# создание последовательностей
seq_df = all_actions.groupby('session_id').agg(
    user=('user', 'first'),
    sequence=('action', list),
    has_anomaly=('is_true_bad', 'max'),
).reset_index()

# добавляем start и end из sessions_df
seq_df = seq_df.merge(
    sessions_df[['session_id', 'start', 'end']],
    on='session_id',
    how='left'
)

seq_df['sequence_length'] = seq_df['sequence'].apply(len)
print(f"\nВсего сессий: {len(seq_df)}")
print(f"Аномальных сессий: {seq_df['has_anomaly'].sum()}")
print(f"Средняя длина: {seq_df['sequence_length'].mean():.1f}")
print(f"Максимальная длина: {seq_df['sequence_length'].max()}")
print(f"Время обработки: {time.time()-start:.1f} секунд")

print("\nПЯТЫЙ ЭТАП: СОХРАНЕНИЕ")
seq_df.to_csv(output_path + "sequences.csv", index=False)
print(f"Сохранено в {output_path}sequences.csv")