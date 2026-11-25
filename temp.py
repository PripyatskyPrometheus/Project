import pandas as pd
extra = pd.read_parquet("D:/dataset/extra_day_features.parquet")
print("Дубли (user, day):", extra.duplicated(subset=['user','day']).sum())  # 0
print("NaN:", extra.isna().sum().sum())  # 0
print("Строк:", len(extra))  # 330452