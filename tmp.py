# analyze_first_subsequent.py
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

RUN_PATH = "D:/dataset/runs/2026-10-29_15-04-49_bilstm/predictions/"
THRESHOLD = None  # если None, подберём на val по user-level F1

def load_preds(run_path, name):
    df = pd.read_parquet(run_path + name + ".parquet")
    if 'day' not in df.columns:
        pkl = pd.read_pickle(f"D:/dataset/{name.replace('_probs','_sequences')}.pkl")
        assert len(df) == len(pkl)
        df = df.reset_index(drop=True)
        df['day'] = pkl['day'].reset_index(drop=True)
        assert (df['user'].values == pkl['user'].values).all()
    return df

def add_first_subsequent(df):
    df = df.sort_values(['user', 'day']).copy()
    anom = df[df['has_anomaly'] == 1].copy()
    first_days = anom.groupby('user')['day'].min().rename('first_anom_day')
    df = df.merge(first_days, on='user', how='left')
    df['is_first'] = (df['has_anomaly'] == 1) & (df['day'] == df['first_anom_day'])
    df['is_subsequent'] = (df['has_anomaly'] == 1) & (df['day'] != df['first_anom_day'])
    return df

def report_group(df, mask_col, threshold, name):
    sub = df[df[mask_col]]
    preds = (sub['prob'] >= threshold).astype(int)
    rec = recall_score(sub['has_anomaly'], preds, zero_division=0)
    prec = precision_score(sub['has_anomaly'], preds, zero_division=0)
    f1 = f1_score(sub['has_anomaly'], preds, zero_division=0)
    caught = int(preds.sum())
    total = len(sub)
    print(f"{name:>20}: caught {caught}/{total}, recall={rec:.4f}, precision={prec:.4f}, f1={f1:.4f}")
    return {'group': name, 'caught': caught, 'total': total, 'recall': rec, 'precision': prec, 'f1': f1}

def main():
    val_df = load_preds(RUN_PATH, "val_probs")
    test_df = load_preds(RUN_PATH, "test_probs")

    val_df = add_first_subsequent(val_df)
    test_df = add_first_subsequent(test_df)

    # если порог не задан — подберём на val по user-level F1 (max prob)
    global THRESHOLD
    if THRESHOLD is None:
        vg = val_df.groupby('user')
        v_probs = vg['prob'].max()
        v_labels = vg['has_anomaly'].max().astype(int)
        best_f1, best_th = 0.0, 0.5
        for th in np.arange(0.05, 0.96, 0.01):
            preds = (v_probs >= th).astype(int)
            f1 = f1_score(v_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, float(th)
        THRESHOLD = best_th
        print(f"Порог по val (user-level F1): {THRESHOLD:.2f} (val F1={best_f1:.4f})")

    print("\n=== TEST: первый vs последующие аномальные дни ===")
    r1 = report_group(test_df, 'is_first', THRESHOLD, 'first_anomalous_day')
    r2 = report_group(test_df, 'is_subsequent', THRESHOLD, 'subsequent_days')
    test_df['is_anomaly'] = (test_df['has_anomaly'] == 1).astype(bool)
    r_all = report_group(test_df, 'is_anomaly', THRESHOLD, 'all_anomalous_days')

    print("\n=== TEST: уровень аккаунта (max prob) ===")
    tg = test_df.groupby('user')
    t_probs = tg['prob'].max()
    t_labels = tg['has_anomaly'].max().astype(int)
    preds = (t_probs >= THRESHOLD).astype(int)
    p = precision_score(t_labels, preds, zero_division=0)
    r = recall_score(t_labels, preds, zero_division=0)
    f1 = f1_score(t_labels, preds, zero_division=0)
    caught = int(preds[t_labels == 1].sum())
    total = int(t_labels.sum())
    print(f"user-level max: caught {caught}/{total}, P={p:.4f}, R={r:.4f}, F1={f1:.4f}")

    out = pd.DataFrame([r1, r2, r_all, {'group': 'user_level_max', 'caught': caught, 'total': total, 'recall': r, 'precision': p, 'f1': f1}])
    out.to_csv("D:/dataset/first_subsequent_report.csv", index=False)
    print("\nСохранено: D:/dataset/first_subsequent_report.csv")

if __name__ == "__main__":
    main()