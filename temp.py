import pandas as pd
df = pd.read_parquet("D:/dataset/extra_day_features_v2.parquet")
print(df[['http','pct_30d_http','delta_30d_http','pc_count','night_share','external']].describe())
print("NaN:", df.isna().sum().sum())