# daily_counters.py
import os
import time
import numpy as np
import pandas as pd
import re

LABELS_DIR = "D:/dataset/labels/"
OUTPUT_PATH = "D:/dataset/extra_day_features.parquet"

INTERNAL_DOMAIN = "dtaa.com"
EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})')


def parse_day_hour(date_series):
    dt = pd.to_datetime(date_series, format='%m/%d/%Y %H:%M:%S', cache=True)
    day = dt.dt.date
    hour = dt.dt.hour
    is_night = ((hour < 7) | (hour >= 22)).astype(np.int32)
    return day, is_night


def process_logon(path):
    print("\n1. logon")
    t0 = time.time()
    df = pd.read_csv(path, usecols=['user', 'date', 'pc', 'activity'])
    df['day'], df['is_night'] = parse_day_hour(df['date'])
    act = df['activity'].astype(str).str.strip().str.lower()
    is_logon = (act == 'logon').astype(np.int32)
    is_logoff = (act == 'logoff').astype(np.int32)

    # pc_count — только уникальные ПК при Logon
    logon_rows = df[is_logon == 1]
    pc_per_day = logon_rows.groupby(['user', 'day'])['pc'].nunique().rename('pc_count').reset_index()

    agg = df.groupby(['user', 'day'], as_index=False).agg(
        logon=('is_night', 'size'),  # временно, заменим ниже
        logon_night=('is_night', 'sum'),
    )
    agg['logon'] = df.groupby(['user', 'day']).size().values
    agg['logoff'] = df.assign(v=is_logoff).groupby(['user', 'day'])['v'].sum().values
    agg = agg.merge(pc_per_day, on=['user', 'day'], how='left').fillna({'pc_count': 0})
    agg['pc_count'] = agg['pc_count'].astype(np.int32)
    print(f"   {time.time()-t0:.1f} сек, строк: {len(agg)}")
    return agg


def process_device(path):
    print("\n2. device")
    t0 = time.time()
    df = pd.read_csv(path, usecols=['user', 'date', 'activity'])
    df['day'], df['is_night'] = parse_day_hour(df['date'])
    is_connect = (df['activity'].astype(str).str.strip().str.lower() == 'connect').astype(np.int32)
    df['is_connect'] = is_connect
    df['usb_night'] = is_connect * df['is_night']
    agg = df.groupby(['user', 'day'], as_index=False).agg(
        usb=('is_connect', 'sum'),
        usb_night=('usb_night', 'sum'),
    )
    print(f"   {time.time()-t0:.1f} сек, строк: {len(agg)}")
    return agg


def process_file(path):
    print("\n3. file")
    t0 = time.time()
    df = pd.read_csv(path, usecols=['user', 'date', 'filename'])
    df['day'], df['is_night'] = parse_day_hour(df['date'])
    agg = df.groupby(['user', 'day'], as_index=False).agg(
        files=('filename', 'size'),
        files_night=('is_night', 'sum'),
    )
    print(f"   {time.time()-t0:.1f} сек, строк: {len(agg)}")
    return agg


def process_email(path):
    print("\n4. email")
    t0 = time.time()
    df = pd.read_csv(path, usecols=['user', 'date', 'to', 'cc', 'bcc', 'attachments'])
    df['day'], df['is_night'] = parse_day_hour(df['date'])
    df['att_num'] = pd.to_numeric(df['attachments'], errors='coerce').fillna(0).astype(np.int32)
    df['has_att'] = (df['att_num'] > 0).astype(np.int32)

    def extract_domains_series(s):
        if pd.isna(s):
            return set()
        return set(m.group(1).lower() for m in EMAIL_RE.finditer(str(s)))

    for col in ('to', 'cc', 'bcc'):
        df[f'domains_{col}'] = df[col].apply(extract_domains_series)


    df['all_domains'] = df.apply(lambda r: r['domains_to'] | r['domains_cc'] | r['domains_bcc'], axis=1)
    df['is_internal'] = df['all_domains'].apply(lambda s: int(len(s) > 0 and s <= {INTERNAL_DOMAIN}))
    df['is_external'] = 1 - df['is_internal']
    df = df.drop(columns=['domains_to', 'domains_cc', 'domains_bcc', 'all_domains'])

    df['ext_night'] = df['is_external'] * df['is_night']

    agg = df.groupby(['user', 'day'], as_index=False).agg(
        email=('att_num', 'size'),
        email_night=('is_night', 'sum'),
        attached=('has_att', 'sum'),
        external=('is_external', 'sum'),
        external_night=('ext_night', 'sum'),
    )
    print(f"   {time.time()-t0:.1f} сек, строк: {len(agg)}")
    return agg


def process_http_chunked(path, chunksize=1_000_000):
    print("\n5. http")
    t0 = time.time()
    parts = []
    for chunk in pd.read_csv(path, usecols=['user', 'date'], chunksize=chunksize):
        day = pd.to_datetime(chunk['date'].str.slice(0, 10), format='%m/%d/%Y').dt.date
        hour = chunk['date'].str.slice(11, 13).astype(np.int32)
        is_night = ((hour < 7) | (hour >= 22)).astype(np.int32)
        tmp = pd.DataFrame({'user': chunk['user'].values, 'day': day.values, 'is_night': is_night.values})
        parts.append(tmp.groupby(['user', 'day'], as_index=False).agg(
            http=('is_night', 'size'),
            http_night=('is_night', 'sum'),
        ))
    agg = pd.concat(parts).groupby(['user', 'day'], as_index=False).sum()
    print(f"   {time.time()-t0:.1f} сек, строк: {len(agg)}")
    return agg


def add_personal_norm(df, columns, window_days=30, min_days=5):
    """Личная норма: rank + delta по календарному окну [день−30, день)."""
    out = df.sort_values(['user', 'day']).reset_index(drop=True)
    values = out[columns].to_numpy(dtype=np.float64)
    dates = out['day'].to_numpy(dtype='datetime64[D]').astype(np.int64)
    n, width = values.shape
    rank = np.full((n, width), 0.5, dtype=np.float32)
    delta = np.zeros((n, width), dtype=np.float32)
    coverage = np.zeros(n, dtype=np.int16)
    gap = np.zeros(n, dtype=np.int16)

    for idx in out.groupby('user', sort=False).indices.values():
        user_dates, user_values = dates[idx], values[idx]
        starts = np.searchsorted(user_dates, user_dates - window_days, side='left')
        for j, start in enumerate(starts):
            row = idx[j]
            coverage[row] = j - start
            if j:
                gap[row] = user_dates[j] - user_dates[j - 1]
            if coverage[row] < min_days:
                continue
            history = user_values[start:j]
            today = user_values[j]
            rank[row] = ((history < today).sum(axis=0) + 0.5 * (history == today).sum(axis=0)) / len(history)
            delta[row] = np.log1p(today) - np.log1p(history).mean(axis=0)

    suffix = f'{window_days}d'
    for j, name in enumerate(columns):
        out[f'pct_{suffix}_{name}'] = rank[:, j]
        out[f'delta_{suffix}_{name}'] = delta[:, j]
    out[f'coverage_{suffix}'] = coverage
    out['previous_observation_gap'] = gap
    return out


def main():
    total = time.time()
    print("ПОЛНЫЕ ДНЕВНЫЕ СЧЁТЧИКИ + ЛИЧНАЯ НОРМА")

    df_logon = process_logon(os.path.join(LABELS_DIR, "logon_labeled.csv"))
    df_device = process_device(os.path.join(LABELS_DIR, "device_labeled.csv"))
    df_file = process_file(os.path.join(LABELS_DIR, "file_labeled.csv"))
    df_email = process_email(os.path.join(LABELS_DIR, "email_labeled.csv"))
    df_http = process_http_chunked(os.path.join(LABELS_DIR, "http_labeled.csv"))

    print("\n6. Merge по (user, day)")
    merged = df_logon
    for part in [df_device, df_file, df_email, df_http]:
        merged = merged.merge(part, on=['user', 'day'], how='outer')
    num_cols = [c for c in merged.columns if c not in ('user', 'day')]
    merged[num_cols] = merged[num_cols].fillna(0)

    # night_share — по всем каналам (logon, logoff, usb, files, http, email)
    total_events = (merged['logon'] + merged['logoff'] + merged['usb'] + merged['files']
                    + merged['http'] + merged['email'])
    night_events = (merged['logon_night'] + merged['usb_night'] + merged['files_night']
                    + merged['http_night'] + merged['email_night'])
    merged['night_share'] = np.where(total_events > 0, night_events / total_events, 0.0).astype(np.float32)
    merged['external_share'] = np.where(merged['email'] > 0, merged['external'] / merged['email'], 0.0).astype(np.float32)
    merged['attachment_share'] = np.where(merged['email'] > 0, merged['attached'] / merged['email'], 0.0).astype(np.float32)
    merged['weekend'] = pd.to_datetime(merged['day']).dt.dayofweek.isin([5, 6]).astype(np.int32)

    # Личная норма (календарное окно 30 дней)
    ROUTINE = ['logon', 'usb', 'files', 'http', 'email', 'external', 'pc_count', 'night_share']
    merged = add_personal_norm(merged, ROUTINE, window_days=30, min_days=5)

    merged = merged.sort_values(['user', 'day']).reset_index(drop=True)

    int_cols = ['logon', 'logon_night', 'logoff', 'pc_count', 'usb', 'usb_night', 'files', 'files_night', 'email', 'email_night', 
    'attached', 'external', 'external_night', 'http', 'http_night', 'weekend']
    merged[int_cols] = merged[int_cols].astype(np.int32)

    merged.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nСохранено: {OUTPUT_PATH}")
    print(f"Размерность: {merged.shape}")
    print(f"Время: {(time.time()-total)/60:.1f} мин")
    print("\nПример:")
    print(merged.head(3).T)

    # df = pd.read_parquet("D:/dataset/extra_day_features_v2.parquet")
    # int_cols = ['logon', 'logon_night', 'logoff', 'pc_count', 'usb', 'usb_night', 'files', 'files_night', 'email', 'email_night', 
    # 'attached', 'external', 'external_night', 'http', 'http_night', 'weekend']
    # df[int_cols] = df[int_cols].astype(np.int32)
    # df.to_parquet("D:/dataset/extra_day_features_v2.parquet", index=False)

if __name__ == '__main__':
    main()