import pandas as pd
df = pd.read_pickle("C:/Users/user/dataset/train_sequences.pkl")
print(df[['files_ratio','file_sensitive_ratio','distinct_pcs_ratio']].describe())
# ожидаем: mean ~0.3-0.5, std > 0.5, max < 7, min 0
print("NaN:", df[['files_ratio','file_sensitive_ratio','distinct_pcs_ratio']].isna().sum().sum())
# ожидаем: 0