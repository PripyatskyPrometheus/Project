# FeatureEngineering.py

import pandas as pd
import time
import os
from pathlib import Path

from labels import load_all_true_bad_ids

def load_insiders(answers_path):

    insiders = pd.read_csv(os.path.join(answers_path, "insiders.csv"))
    insiders_df = insiders[insiders['dataset'] == 4.2].copy()
    
    print(f"Инсайдеры в наборе данных CERT r4.2")
    print(f"Количество: {len(insiders_df)}")
    
    return insiders_df

def get_insider_labels(insiders_df):
    insiders_df['start'] = pd.to_datetime(insiders_df['start'])
    insiders_df['end'] = pd.to_datetime(insiders_df['end'])
    
    labels = {}
    for _, row in insiders_df.iterrows():
        labels[row['user']] = {
            'scenario': row['scenario'],
            'details': row['details'],
            'start': row['start'],
            'end': row['end']
        }
    
    print(f"Загружено {len(labels)} инсайдеров")
    return labels

def process_logon_file(labeled_file_path):
    print("\nАГРЕГАЦИЯ LOGON")
    start = time.time()
    
    df = pd.read_csv(labeled_file_path, usecols=['user', 'date', 'pc', 'activity', 'is_true_bad'])
    df['parsed_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S')
    df['day'] = df['parsed_date'].dt.date
    df['night'] = df['parsed_date'].dt.hour.apply(lambda x: 1 if x <= 6 or x >= 23 else 0)
    df['weekend'] = df['parsed_date'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    df['is_logon'] = df['activity'].str.contains('Logon', na=False).astype(int)
    df['is_logoff'] = df['activity'].str.contains('Logoff', na=False).astype(int)
    
    result = df.groupby(['user', 'day']).agg(
        logon_total=('activity', 'count'),
        logon_unique_pcs=('pc', 'nunique'),
        logon_night=('night', 'sum'),
        logon_weekend=('weekend', 'sum'),
        logons=('is_logon', 'sum'),
        logoffs=('is_logoff', 'sum'),
        logon_bad=('is_true_bad', 'sum'),
        logon_has_bad=('is_true_bad', 'max')
    ).reset_index()
    
    print(f"  Время: {time.time()-start} сек | {len(result)} записей")
    return result

def save_checkpoint(df, filename, output_dir="D:/dataset/"):
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Сохранено: {filepath} ({len(df)} записей)")
    return filepath

def process_device_file(labeled_file_path):
    print("\nАГРЕГАЦИЯ DEVICE")
    start = time.time()
    
    df = pd.read_csv(labeled_file_path, usecols=['user', 'date', 'pc', 'activity', 'is_true_bad'])
    df['parsed_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S')
    df['day'] = df['parsed_date'].dt.date
    df['night'] = df['parsed_date'].dt.hour.apply(lambda x: 1 if x <= 6 or x >= 23 else 0)
    df['weekend'] = df['parsed_date'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    df['is_connect'] = df['activity'].str.contains('Connect', na=False).astype(int)
    df['is_disconnect'] = df['activity'].str.contains('Disconnect', na=False).astype(int)
    
    result = df.groupby(['user', 'day']).agg(
        device_total=('activity', 'count'),
        device_unique_pcs=('pc', 'nunique'),
        device_night=('night', 'sum'),
        device_weekend=('weekend', 'sum'),
        device_connects=('is_connect', 'sum'),
        device_disconnects=('is_disconnect', 'sum'),
        device_bad=('is_true_bad', 'sum'),
        device_has_bad=('is_true_bad', 'max')
    ).reset_index()
    
    print(f"  Время: {time.time()-start} сек | {len(result)} записей")
    return result

def merge_two_features(df_left, df_right, left_name, right_name):
    print(f"\nОбъединение {left_name} и {right_name}...")
    
    df_left['day'] = df_left['day'].astype(str)
    df_right['day'] = df_right['day'].astype(str)
    
    all_users_days = pd.concat([
        df_left[['user', 'day']],
        df_right[['user', 'day']]
    ]).drop_duplicates().reset_index(drop=True)
    
    print(f"Всего комбинаций: {len(all_users_days)}")
    
    merged = all_users_days.merge(df_left, on=['user', 'day'], how='left')
    merged = merged.merge(df_right, on=['user', 'day'], how='left')
    
    merged = merged.fillna(0)
    
    return merged

def process_email_file(labeled_file_path):
    print("\nАГРЕГАЦИЯ EMAIL")
    start = time.time()
    
    df = pd.read_csv(labeled_file_path, usecols=['user', 'date', 'to', 'cc', 'bcc', 'size', 'attachments', 'is_true_bad'])
    df['parsed_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S')
    df['day'] = df['parsed_date'].dt.date
    df['night'] = df['parsed_date'].dt.hour.apply(lambda x: 1 if x <= 6 or x >= 23 else 0)
    df['weekend'] = df['parsed_date'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    df['has_attachment'] = (df['attachments'] > 0).astype(int)
    df['is_external'] = df['to'].apply(lambda x: 1 if pd.notna(x) and 'dtaa.com' not in str(x) else 0)
    df['recipients_count'] = df['to'].apply(lambda x: len(str(x).split(';')) if pd.notna(x) else 0)
    
    result = df.groupby(['user', 'day']).agg(
        email_total=('date', 'count'),
        email_night=('night', 'sum'),
        email_weekend=('weekend', 'sum'),
        email_with_attachments=('has_attachment', 'sum'),
        total_attachments=('attachments', 'sum'),
        email_external=('is_external', 'sum'),
        avg_email_size=('size', 'mean'),
        max_email_size=('size', 'max'),
        total_email_size=('size', 'sum'),
        avg_recipients=('recipients_count', 'mean'),
        max_recipients=('recipients_count', 'max'),
        email_bad=('is_true_bad', 'sum'),
        email_has_bad=('is_true_bad', 'max')
    ).reset_index()
    
    print(f"  Время: {time.time()-start} сек | {len(result)} записей")
    return result

def process_file_file(file_path):
    print("\nАГРЕГАЦИЯ FILE")
    start = time.time()
    
    df = pd.read_csv(file_path, usecols=['user', 'date', 'filename', 'is_true_bad'])
    df['parsed_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S')
    df['day'] = df['parsed_date'].dt.date
    df['night'] = df['parsed_date'].dt.hour.apply(lambda x: 1 if x <= 6 or x >= 23 else 0)
    df['weekend'] = df['parsed_date'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    
    # Извлекаем расширение
    def get_extension(fname):
        if pd.isna(fname):
            return 'no_extension'
        parts = str(fname).split('.')
        return parts[-1].lower() if len(parts) > 1 else 'no_extension'
    
    df['ext'] = df['filename'].apply(get_extension)
    df['sensitive'] = df['ext'].apply(lambda x: 1 if x in ['doc', 'docx', 'xls', 'xlsx', 'pdf', 'ppt', 'pptx', 'txt', 'rtf', 'csv', 'sql'] else 0)
    df['archive'] = df['ext'].apply(lambda x: 1 if x in ['zip', 'rar', '7z', 'tar', 'gz'] else 0)
    df['executable'] = df['ext'].apply(lambda x: 1 if x in ['exe', 'msi', 'bat', 'cmd', 'ps1', 'sh'] else 0)
    
    result = df.groupby(['user', 'day']).agg(
        file_total=('filename', 'count'),
        file_night=('night', 'sum'),
        file_weekend=('weekend', 'sum'),
        sensitive_files=('sensitive', 'sum'),
        archive_files=('archive', 'sum'),
        executable_files=('executable', 'sum'),
        file_bad=('is_true_bad', 'sum'),
        file_has_bad=('is_true_bad', 'max')
    ).reset_index()
    
    print(f"  Время: {time.time()-start} сек | {len(result)} записей")
    return result

def add_ldap_features(features_df, ldap_path):
    print("\nДОБАВЛЕНИЕ LDAP ИНФОРМАЦИИ")
    
    # Загружаем последний LDAP-файл
    ldap_files = sorted(Path(ldap_path).glob("*.csv"))
    latest_ldap = pd.read_csv(ldap_files[-1])
    print(f"Используем файл: {ldap_files[-1].name}")
    print(f"Колонки в LDAP: {list(latest_ldap.columns)}")
    
    latest_ldap = latest_ldap.rename(columns={'user_id': 'user'})
    useful_columns = ['user'] 
    
    for col in latest_ldap.columns:
        if col not in ['user', 'employee_name', 'email']:
            useful_columns.append(col)
    
    ldap_subset = latest_ldap[useful_columns].copy()
    print(f"Используем колонки: {useful_columns}")
    
    features_with_ldap = features_df.merge(ldap_subset, on='user', how='left')
    
    print(f"Размер до: {features_df.shape}")
    print(f"Размер после: {features_with_ldap.shape}")
    print(f"Добавлено колонок: {len(useful_columns) - 1}")
    
    # Проверяем, всем ли пользователям нашлась роль
    if 'role' in features_with_ldap.columns:
        missing_roles = features_with_ldap[features_with_ldap['role'].isna()]['user'].nunique()
        print(f"Пользователей без роли: {missing_roles}")
    
    return features_with_ldap

def add_insider_labels(df, labels_dict):
    """Добавляет колонку is_insider ( ставим метку "1" если пользователь был инсайдером в этот день)"""
    df['is_insider'] = 0
    df['scenario'] = 0
    
    for idx, row in df.iterrows():
        user = row['user']
        day = pd.to_datetime(row['day'])
        
        if user in labels_dict:
            insider = labels_dict[user]
            if insider['start'] <= day <= insider['end']:
                df.at[idx, 'is_insider'] = 1
                df.at[idx, 'scenario'] = insider['scenario']
    
    return df

def process_http_file(file_path):
    print("\nАГРЕГАЦИЯ HTTP...")
    start = time.time()
    
    df = pd.read_csv(file_path, usecols=['user', 'date', 'url', 'is_true_bad'])
    df['parsed_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S')
    df['day'] = df['parsed_date'].dt.date
    df['night'] = df['parsed_date'].dt.hour.apply(lambda x: 1 if x <= 6 or x >= 23 else 0)
    df['weekend'] = df['parsed_date'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    
    def get_category(url):
        url_lower = str(url).lower()
        if any(x in url_lower for x in ['facebook', 'twitter', 'linkedin', 'instagram']):
            return 'social'
        elif any(x in url_lower for x in ['dropbox', 'drive.google', 'mega', 'cloud']):
            return 'cloud'
        elif any(x in url_lower for x in ['job', 'career', 'indeed', 'monster']):
            return 'job'
        elif any(x in url_lower for x in ['wikileaks', 'leak', 'anonymous']):
            return 'leak'
        else:
            return 'other'
    
    df['category'] = df['url'].apply(get_category)
    
    result = df.groupby(['user', 'day']).agg(
        http_total=('url', 'count'),
        http_night=('night', 'sum'),
        http_weekend=('weekend', 'sum'),
        social_media=('category', lambda x: (x == 'social').sum()),
        cloud_storage=('category', lambda x: (x == 'cloud').sum()),
        job_search=('category', lambda x: (x == 'job').sum()),
        leak_site=('category', lambda x: (x == 'leak').sum()),
        http_bad=('is_true_bad', 'sum'),
        http_has_bad=('is_true_bad', 'max')
    ).reset_index()
    
    print(f"  Время: {time.time()-start} сек | {len(result)} записей")
    return result

def add_psychometric_features(features_df, psycho_path):
    """
    Добавляет психометрические данные к фичам
    """
    print("\nДОБАВЛЕНИЕ PSYCHOMETRIC ДАННЫХ")
    
    psycho_df = pd.read_csv(psycho_path)
    
    # В psychometric.csv обычно колонки: user, O, C, E, A, N
    if 'user_id' in psycho_df.columns:
        psycho_df = psycho_df.rename(columns={'user_id': 'user'})
    
    features_with_psycho = features_df.merge(psycho_df, on='user', how='left')
    
    print(f"Размер до: {features_df.shape}")
    print(f"Размер после: {features_with_psycho.shape}")

    missing = features_with_psycho['O'].isna().sum() if 'O' in features_with_psycho.columns else 0
    print(f"Пользователей без психометрики: {missing}")
    
    return features_with_psycho

if __name__ == "__main__":
    input_path = "D:/НИРС/cert4.2/r4.2"
    answers_path = "D:/НИРС/answers/"
    output_path = "D:/dataset/"
    
    # Загружаем метки инсайдеров (для дневных меток)
    insiders_df = load_insiders(answers_path)
    insider_labels = get_insider_labels(insiders_df)
    
    # Загружаем точные ID плохих действий
    true_bad_ids = load_all_true_bad_ids(answers_path)
    all_bad_ids = set()
    for user, types in true_bad_ids.items():
        for atype, ids in types.items():
            all_bad_ids.update(ids)
    
    # Агрегируем каждый размеченный файл
    logon_features = process_logon_file("D:/dataset/labels/logon_labeled.csv")
    save_checkpoint(logon_features, "logon_features.csv", output_path)
    
    device_features = process_device_file("D:/dataset/labels/device_labeled.csv")
    save_checkpoint(device_features, "device_features.csv", output_path)
    
    email_features = process_email_file("D:/dataset/labels/email_labeled.csv")
    save_checkpoint(email_features, "email_featuresw.csv", output_path)
    
    file_features = process_file_file("D:/dataset/labels/file_labeled.csv")
    save_checkpoint(file_features, "file_features.csv", output_path)
    
    http_features = process_http_file("D:/dataset/labels/http_labeled.csv")
    save_checkpoint(http_features, "http_features.csv", output_path)
    
    # Объединяем все признаки
    combined = logon_features.copy()
    combined = merge_two_features(combined, device_features, "logon", "device")
    combined = merge_two_features(combined, email_features, "logon_device", "email")
    combined = merge_two_features(combined, file_features, "logon_device_email", "file")
    combined = merge_two_features(combined, http_features, "logon_device_email_file", "http")
    
    # Добавляем LDAP
    combined = add_ldap_features(combined, os.path.join(input_path, "LDAP"))

    # Добавляем в пайплайн после LDAP
    combined = add_psychometric_features(combined, os.path.join(input_path, "psychometric.csv"))

    save_checkpoint(combined, "features_final_with_psycho.csv", output_path)

    # После добавления LDAP и психометрики
    domain_features = pd.read_csv("D:/dataset/domain_features.csv")

    # Объединяем
    combined = combined.merge(domain_features, on='user', how='left')

    # Заполняем пропуски нулями
    combined = combined.fillna(0)

    print(f"\nПосле добавления доменных признаков: {combined.shape}")

    # Добавляем дневные метки инсайдеров (для сравнения)
    combined = add_insider_labels(combined, insider_labels)
    
    # Сохраняем финальный датасет
    save_checkpoint(combined, "features_final.csv", output_path)
    
    print(f"\nФИНАЛЬНЫЙ ДАТАСЕТ ГОТОВ!")
    print(f"Размер: {combined.shape}")
    print(f"Колонки: {list(combined.columns)}")
    print(f"Всего записей: {len(combined)}")
    print(f"Инсайдерских записей (по старым меткам): {combined['is_insider'].sum()}")
    print(f"Записей с реальными плохими действиями: {(combined['logon_has_bad'] + combined['device_has_bad']
            + combined['email_has_bad'] + combined['file_has_bad'] + combined['http_has_bad'] > 0).sum()}")