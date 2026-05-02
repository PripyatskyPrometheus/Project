import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

df = pd.read_csv("D:/dataset/features_final.csv")

print("ОБЩАЯ ИНФОРМАЦИЯ:")
print(f"Размер: {df.shape}")
print(f"\nКолонки: {list(df.columns)}")
print(f"\nТипы данных:")
print(df.dtypes.value_counts())
print(f"\nколичество признаков: {len(df.columns)}")

print("\nПЕРВЫЕ СТРОКИ:")
print(df.head())

print("\nСТАТИСТИКА ПО ЧИСЛОВЫМ ПРИЗНАКАМ:")
print(df.describe())

print("\nПроверяем ПРОПУСКИ (MISSING VALUES)")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({'missing': missing, 'percent': missing_percent})
missing_df = missing_df[missing_df['missing'] > 0].sort_values('percent', ascending=False)

if len(missing_df) > 0:
    print(missing_df)
else:
    print("Пропусков нет!")

msno.matrix(df)
plt.title("Матрица пропусков")
plt.tight_layout()
plt.savefig('plots/missing_matrix.png', dpi=150)
plt.close()

print("\nПРОВЕРКА ДУБЛИКАТОВ:")
print(f"Полных дубликатов строк: {df.duplicated().sum()}")

key_duplicates = df.duplicated(subset=['user', 'day']).sum()
print(f"Дубликатов по ключу (user, day): {key_duplicates}")


df['has_anomaly'] = (
    (df['logon_bad'] > 0) |
    (df['device_bad'] > 0) |
    (df['http_bad'] > 0) |
    (df['email_bad'] > 0) |
    (df['file_bad'] > 0)
).astype(int)

print("\nРАСПРЕДЕЛЕНИЕ ТАРГЕТА:")
print(df['has_anomaly'].value_counts())
print(f"Процент аномалий: {df['has_anomaly'].mean()*100}%")

print("\nРАСПРЕДЕЛЕНИЕ АНОМАЛИЙ ПО ПОЛЬЗОВАТЕЛЯМ:")

# Считаем количество аномальных дней на пользователя
anomaly_by_user = df[df['has_anomaly'] == 1].groupby('user').size().sort_values(ascending=False)

print(f"Всего пользователей с аномалиями: {len(anomaly_by_user)}")
print(f"Максимум аномальных дней у одного пользователя: {anomaly_by_user.max()}")
print(f"Среднее: {anomaly_by_user.mean()}")
print(f"Медиана: {anomaly_by_user.median()}")

print("\nGользователи с аномальными днями:")
print(anomaly_by_user)

plt.figure(figsize=(10, 5))
anomaly_by_user.hist(bins=30)
plt.xlabel("Количество аномальных дней")
plt.ylabel("Количество пользователей")
plt.title("Распределение аномалий по пользователям")
plt.tight_layout()
plt.savefig('plots/anomaly_by_user_distribution.png', dpi=150)
plt.close()


print("\nКОРРЕЛЯЦИЯ ПРИЗНАКОВ С ТАРГЕТОМ ===")

# Список признаков для корреляции (исключаем явные утечки и нечисловые)
exclude_corr = ['user', 'day', 'is_insider', 'scenario', 'employee_name', 'top_domain','role', 'business_unit', 'functional_unit', 'department', 'team', 'supervisor',
                'logon_bad', 'device_bad', 'http_bad', 'email_bad', 'file_bad',
                'logon_has_bad', 'device_has_bad', 'http_has_bad', 'email_has_bad', 'file_has_bad',
                'bad_http', 'device_total', 'device_unique_pcs', 'device_connects', 'device_disconnects',
                'device_night', 'device_weekend', 'device_has_bad']

# Берём числовые колонки, исключая утечки
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_cols = [col for col in numeric_cols if col not in exclude_corr and col != 'has_anomaly']

# Считаем корреляцию с таргетом
corr_with_target = df[corr_cols + ['has_anomaly']].corr()['has_anomaly'].drop('has_anomaly').sort_values(ascending=False)

print("\nТоп-15 признаков с наибольшей положительной корреляцией с аномалией:")
print(corr_with_target.head(15))

print("\nТоп-5 признаков с наибольшей отрицательной корреляцией с аномалией:")
print(corr_with_target.tail(5))

corr_with_target.head(10).to_csv('plots/corr_with_target.csv')
print("\nКорреляции сохранены в plots/corr_with_target.csv")


print("\nТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИЙ ===")

# Берём топ-20 признаков по корреляции с таргетом
top_features = corr_with_target.head(20).index.tolist()
corr_matrix = df[top_features + ['has_anomaly']].corr()

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Тепловая карта корреляций (топ-20 признаков по связи с аномалией)", fontsize=14)
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=150)
plt.close()
print("Тепловая карта сохранена в plots/correlation_heatmap.png")


print("\nBOXPLOT: СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ (НОРМА vs АНОМАЛИЯ)")

# Выбираем топ-6 признаков для визуализации
top6_features = corr_with_target.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feat in enumerate(top6_features):
    data_to_plot = [df[df['has_anomaly'] == 0][feat].dropna(),
                    df[df['has_anomaly'] == 1][feat].dropna()]
    
    bp = axes[i].boxplot(data_to_plot, labels=['Норма', 'Аномалия'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('salmon')
    axes[i].set_title(feat, fontsize=10)
    axes[i].set_ylabel('Значение')
    axes[i].grid(True, alpha=0.3)

plt.suptitle("Сравнение распределений признаков: нормальные дни vs аномальные дни", fontsize=14)
plt.tight_layout()
plt.savefig('plots/boxplot_top_features.png', dpi=150)
plt.close()
print("Boxplot сохранён в plots/boxplot_top_features.png")


print("\nАНАЛИЗ ВЫБРОСОВ В КЛЮЧЕВЫХ ПРИЗНАКАХ")

for feat in top6_features:
    Q1 = df[feat].quantile(0.25)
    Q3 = df[feat].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[feat] < lower_bound) | (df[feat] > upper_bound)]
    
    print(f"\n{feat}:")
    print(f"  Нижняя граница: {lower_bound}")
    print(f"  Верхняя граница: {upper_bound}")
    print(f"  Количество выбросов: {len(outliers)} ({len(outliers)/len(df)*100:}%)")
    
    # Для признаков с большим количеством выбросов показываем примеры
    if len(outliers) > 0 and len(outliers) < 100:
        print(f"  Примеры значений выбросов: {outliers[feat].head(3).tolist()}")


print(f"Все графики сохранены в папке 'plots/'")