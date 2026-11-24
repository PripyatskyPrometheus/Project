# build_extra_features.py
import os
import time
from collections import Counter
import numpy as np
import pandas as pd

LABELS_DIR = "D:/dataset/labels/"
OUTPUT_PATH = "D:/dataset/extra_day_features.parquet"


def extract_day_and_night(date_series):
    """Быстро извлекает дату (datetime.date) и флаг ночного времени (<7 или >=22). Формат MM/DD/YYYY HH:MM:SS"""
    dt = pd.to_datetime(date_series, format='%m/%d/%Y %H:%M:%S', cache=True)
    day = dt.dt.date
    hour = dt.dt.hour
    is_night = ((hour < 7) | (hour >= 22)).astype(np.int32)
    return day, is_night


def process_logon(path):
    print("\n1. Обработка logon_labeled.csv")
    t0 = time.time()
    df = pd.read_csv(path, usecols=['user', 'date', 'pc', 'activity'])
    df['day'], df['is_night'] = extract_day_and_night(df['date'])

    agg = df.groupby(['user', 'day'], as_index=False).agg(
        distinct_pcs_real=('pc', 'nunique'),
        n_logon_night=('is_night', 'sum'),
        n_logon_total=('activity', 'count'),
    )
    print(f"   Готово за {time.time() - t0:.1f} сек. Строк: {len(agg)}")
    return agg


def process_device(path):
    print("\n2. Обработка device_labeled.csv")
    t0 = time.time()
    df = pd.read_csv(path, usecols=['user', 'date', 'activity'])
    df['day'], df['is_night'] = extract_day_and_night(df['date'])
    is_connect = (df['activity'].astype(str).str.strip().str.lower() == 'connect').astype(np.int32)
    df['is_connect'] = is_connect
    df['usb_night'] = is_connect * df['is_night']

    agg = df.groupby(['user', 'day'], as_index=False).agg(
        n_usb_connect=('is_connect', 'sum'),
        n_usb_night=('usb_night', 'sum'),
    )
    print(f"   Готово за {time.time() - t0:.1f} сек. Строк: {len(agg)}")
    return agg


def process_file(path):
    print("\n3. Обработка file_labeled.csv")
    t0 = time.time()
    df = pd.read_csv(path, usecols=['user', 'date', 'filename'])
    df['day'], df['is_night'] = extract_day_and_night(df['date'])

    agg = df.groupby(['user', 'day'], as_index=False).agg(
        n_file_total=('filename', 'count'),
        n_file_night=('is_night', 'sum'),
    )
    print(f"   Готово за {time.time() - t0:.1f} сек. Строк: {len(agg)}")
    return agg


def process_email(path):
    print("\n4. Обработка email_labeled.csv")
    t0 = time.time()
    df = pd.read_csv(path, usecols=['user', 'date', 'attachments'])
    df['day'], df['is_night'] = extract_day_and_night(df['date'])
    df['att_num'] = pd.to_numeric(df['attachments'], errors='coerce').fillna(0).astype(np.int32)

    agg = df.groupby(['user', 'day'], as_index=False).agg(
        n_email_total=('att_num', 'count'),
        total_attachments=('att_num', 'sum'),
        n_email_night=('is_night', 'sum'),
    )
    print(f"   Готово за {time.time() - t0:.1f} сек. Строк: {len(agg)}")
    return agg


def process_http_chunked(path, chunksize=1_000_000):
    print("\n5. Обработка http_labeled.csv")
    t0 = time.time()
    user_day_counters = {}
    night_counts = Counter()
    risk_counts = Counter()

    risk_pattern = r'wikileaks|leak|anonymous|dropbox|drive\.google|mega|cloud|keylog|spy|hack|job|career|indeed|monster'

    for i, chunk in enumerate(pd.read_csv(path, usecols=['user', 'date', 'url'], chunksize=chunksize), 1):
        # Быстрый парсинг даты и часа по срезу строки MM/DD/YYYY HH:MM:SS
        day = pd.to_datetime(chunk['date'].str.slice(0, 10), format='%m/%d/%Y').dt.date
        hour = chunk['date'].str.slice(11, 13).astype(np.int32)
        is_night = ((hour < 7) | (hour >= 22)).astype(np.int32)

        url_lower = chunk['url'].astype(str).str.lower()
        is_risk = url_lower.str.contains(risk_pattern, regex=True, na=False).astype(np.int32)

        # Извлекаем чистый домен без протокола, пути и порта
        domains = (
            url_lower
            .str.replace(r'^https?://', '', regex=True)
            .str.split('/', n=1).str[0]
            .str.split(':', n=1).str[0]
        )

        chunk_df = pd.DataFrame({
            'user': chunk['user'].values,
            'day': day.values,
            'domain': domains.values,
            'is_night': is_night.values,
            'is_risk': is_risk.values,
        })

        # Агрегация ночных и рисковых запросов по (user, day)
        nr_grouped = chunk_df.groupby(['user', 'day'])[['is_night', 'is_risk']].sum()
        for key, row in nr_grouped.iterrows():
            if row['is_night'] > 0:
                night_counts[key] += int(row['is_night'])
            if row['is_risk'] > 0:
                risk_counts[key] += int(row['is_risk'])

        # Подсчёт частот доменов по (user, day)
        dom_grouped = chunk_df.groupby(['user', 'day', 'domain']).size()
        for (user, d, dom), cnt in dom_grouped.items():
            key = (user, d)
            if key not in user_day_counters:
                user_day_counters[key] = Counter()
            user_day_counters[key][dom] += int(cnt)

        if i % 5 == 0:
            print(f"   Обработано {i * chunksize:,} строк HTTP ({time.time() - t0:.1f} сек)...")

    print("   Вычисление энтропии Шеннона для каждого (user, day)...")
    rows = []
    for (user, d), counter in user_day_counters.items():
        counts = np.array(list(counter.values()), dtype=np.float64)
        total = int(counts.sum())
        k = len(counts)

        if total == 0 or k <= 1:
            entropy = 0.0
            norm_entropy = 0.0
        else:
            probs = counts / total
            entropy = float(-np.sum(probs * np.log2(probs)))
            norm_entropy = float(entropy / np.log2(k))

        key = (user, d)
        rows.append({
            'user': user,
            'day': d,
            'n_http_total': total,
            'unique_domains': k,
            'domain_entropy': entropy,
            'domain_entropy_norm': norm_entropy,
            'n_http_night': night_counts.get(key, 0),
            'n_http_risk': risk_counts.get(key, 0),
        })

    agg = pd.DataFrame(rows)
    print(f"   HTTP готов за {time.time() - t0:.1f} сек. Строк: {len(agg)}")
    return agg


def main():
    total_start = time.time()
    print("ИЗВЛЕЧЕНИЕ РАСШИРЕННЫХ ДНЕВНЫХ ПРИЗНАКОВ")

    df_logon = process_logon(os.path.join(LABELS_DIR, "logon_labeled.csv"))
    df_device = process_device(os.path.join(LABELS_DIR, "device_labeled.csv"))
    df_file = process_file(os.path.join(LABELS_DIR, "file_labeled.csv"))
    df_email = process_email(os.path.join(LABELS_DIR, "email_labeled.csv"))
    df_http = process_http_chunked(os.path.join(LABELS_DIR, "http_labeled.csv"))

    print("\n6. Объединение всех источников по (user, day)")
    merged = df_logon
    for part in [df_device, df_file, df_email, df_http]:
        merged = pd.merge(merged, part, on=['user', 'day'], how='outer')

    # Заполняем нулями дни, когда в каком-то канале не было событий
    num_cols = [c for c in merged.columns if c not in ('user', 'day')]
    merged[num_cols] = merged[num_cols].fillna(0)

    int_cols = [c for c in num_cols if c not in ('domain_entropy', 'domain_entropy_norm')]
    for c in int_cols:
        merged[c] = merged[c].astype(np.int32)

    # Признак выходного дня (5 = суббота, 6 = воскресенье)
    day_dt = pd.to_datetime(merged['day'])
    merged['is_weekend'] = (day_dt.dt.dayofweek >= 5).astype(np.int32)

    # Доля ночной активности от всех событий дня
    total_events = (
        merged['n_logon_total']
        + merged['n_usb_connect']
        + merged['n_file_total']
        + merged['n_email_total']
        + merged['n_http_total']
    )
    night_events = (
        merged['n_logon_night']
        + merged['n_usb_night']
        + merged['n_file_night']
        + merged['n_email_night']
        + merged['n_http_night']
    )
    merged['night_activity_ratio'] = np.where(total_events > 0, night_events / total_events, 0.0).astype(np.float32)

    merged = merged.sort_values(['user', 'day']).reset_index(drop=True)
    merged.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nСохранено в: {OUTPUT_PATH}")
    print(f"Итоговая размерность: {merged.shape}")
    print(f"Общее время работы: {(time.time() - total_start) / 60:.2f} мин.")
    print("\nПример данных:")
    print(merged.head(3).T)


if __name__ == '__main__':
    main()
