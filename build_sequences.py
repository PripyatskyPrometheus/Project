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
    print(f"\n{number}. Загрузка {type_name}_labeled.csv...")
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

for name, df in [('logon', logon), ('device', device), ('email', email), ('file', file), ('http', http)]:
    df['parsed_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S')
    df.drop('date', axis=1, inplace=True)
    print(f"{name}: даты преобразованы")


print("\nТРЕТИЙ ЭТАП: ПОИСК СЕССИЙ")

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
                'pc': current_logon['pc']
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
            'pc': current_logon['pc']
        })
        current_logon = None

sessions_df = pd.DataFrame(sessions)
print(f"Всего сессий: {len(sessions_df)}")
print(f"Уникальных пользователей: {sessions_df['user'].nunique()}")

print("\nЧЕТВЁРТЫЙ ЭТАП: СБОР ДЕЙСТВИЙ ВНУТРИ СЕССИЙ")

def get_actions_in_session(session, device, http, file, email):
    user = session['user']
    start = session['start']
    end = session['end']
    
    actions = []
    
    actions.append(('logon', start, 0))
    
    d = device[(device['user'] == user) & 
               (device['parsed_date'] >= start) & 
               (device['parsed_date'] <= end)]
    for _, row in d.iterrows():
        act = 'device_connect' if 'Connect' in str(row['activity']) else 'device_disconnect'
        actions.append((act, row['parsed_date'], row['is_true_bad']))
    
    h = http[(http['user'] == user) & 
             (http['parsed_date'] >= start) & 
             (http['parsed_date'] <= end)]
    for _, row in h.iterrows():
        # Категоризируем URL
        url_lower = str(row['url']).lower()
        if any(x in url_lower for x in ['wikileaks', 'leak', 'anonymous']):
            act = 'http_leak'
        elif any(x in url_lower for x in ['job', 'career', 'indeed', 'monster']):
            act = 'http_job'
        elif any(x in url_lower for x in ['facebook', 'twitter', 'linkedin', 'instagram']):
            act = 'http_social'
        elif any(x in url_lower for x in ['dropbox', 'drive.google', 'mega', 'cloud']):
            act = 'http_cloud'
        elif any(x in url_lower for x in ['gmail', 'mail', 'outlook']):
            act = 'http_email_service'
        elif any(x in url_lower for x in ['keylog', 'spy', 'hack']):
            act = 'http_hack'
        else:
            act = 'http_other'
        actions.append((act, row['parsed_date'], row['is_true_bad']))
    
    f = file[(file['user'] == user) & 
             (file['parsed_date'] >= start) & 
             (file['parsed_date'] <= end)]
    for _, row in f.iterrows():
        fname = str(row['filename'])
        ext = fname.split('.')[-1].lower() if '.' in fname else 'other'
        if ext in ['doc', 'docx', 'xls', 'xlsx', 'pdf', 'ppt', 'pptx', 'txt', 'rtf', 'csv', 'sql']:
            act = 'file_sensitive'
        elif ext in ['zip', 'rar', '7z', 'tar', 'gz']:
            act = 'file_archive'
        elif ext in ['exe', 'msi', 'bat', 'cmd', 'ps1', 'sh']:
            act = 'file_executable'
        elif ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            act = 'file_image'
        else:
            act = 'file_other'
        actions.append((act, row['parsed_date'], row['is_true_bad']))
    
    e = email[(email['user'] == user) & 
              (email['parsed_date'] >= start) & 
              (email['parsed_date'] <= end)]
    for _, row in e.iterrows():
        # Внешний или внутренний
        if 'dtaa.com' not in str(row['to']):
            act = 'email_external'
        else:
            act = 'email_internal'
        actions.append((act, row['parsed_date'], row['is_true_bad']))
        
        # Вложение
        if row['attachments'] > 0:
            actions.append(('email_attach', row['parsed_date'], row['is_true_bad']))
        
        # Ночное письмо
        hour = row['parsed_date'].hour
        if hour <= 6 or hour >= 23:
            actions.append(('email_night', row['parsed_date'], row['is_true_bad']))
    
    actions.append(('logoff', end, 0))
    
    # Сортируем по времени
    actions.sort(key=lambda x: x[1])
    
    return actions

# Обрабатываем первые 1000 сессий для теста
print("Обработка сессий... (первые 1000 для теста)")
test_sessions = sessions_df.head(1000)
session_sequences = []

for i, session in test_sessions.iterrows():
    if i % 100 == 0:
        print(f"  Обработано {i} сессий...")
    
    actions = get_actions_in_session(session, device, http, file, email)
    sequence = [a[0] for a in actions]
    has_anomaly = 1 if any(a[2] == 1 for a in actions) else 0
    
    session_sequences.append({
        'user': session['user'],
        'start': session['start'],
        'end': session['end'],
        'sequence': sequence,
        'sequence_length': len(sequence),
        'has_anomaly': has_anomaly
    })

seq_df = pd.DataFrame(session_sequences)
print(f"\nВсего сессий: {len(seq_df)}")
print(f"Аномальных сессий: {seq_df['has_anomaly'].sum()}")

print("ПЯТЫЙ ЭТАП: СОХРАНЕНИЕ")
seq_df.to_csv(output_path + "sequences_test.csv", index=False)
print(f"Сохранено в {output_path}sequences_test.csv")