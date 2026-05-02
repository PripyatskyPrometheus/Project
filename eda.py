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


print("\nКОРРЕЛЯЦИЯ ПРИЗНАКОВ С ТАРГЕТОМ")

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


print("\nТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИЙ")

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


print("\nАНАЛИЗ ТОП-ПОЛЬЗОВАТЕЛЕЙ С АНОМАЛИЯМИ (ПРОВЕРКА НА ПЕРЕОБУЧЕНИЕ):")

# Берём топ-5 пользователей с наибольшим числом аномалий
top_users = anomaly_by_user.head(5).index.tolist()
print(f"Топ-5 пользователей: {top_users}")

for user in top_users:
    user_data = df[df['user'] == user]
    total_days = len(user_data)
    anomaly_days = user_data['has_anomaly'].sum()
    print(f"\n{user}:")
    print(f"  Всего дней: {total_days}")
    print(f"  Аномальных дней: {anomaly_days} ({anomaly_days/total_days*100}%)")
    
    # Смотрим, какие признаки у этого пользователя выше среднего
    for feat in ['sensitive_files', 'job_search', 'leak_site', 'http_night']:
        user_mean = user_data[feat].mean()
        global_mean = df[feat].mean()
        if user_mean > global_mean * 1.5:
            print(f"  {feat}: {user_mean} (глобально: {global_mean})")

print("\nСРАВНЕНИЕ НОРМИРОВАННЫХ И ИСХОДНЫХ ПРИЗНАКОВ:")

# Нормированные признаки в датасете
ratio_features = ['sensitive_ratio', 'logon_night_ratio', 'http_night_ratio', 
                  'email_night_ratio', 'device_night_ratio', 'external_ratio', 'attachment_ratio']

for ratio_feat in ratio_features:
    if ratio_feat in df.columns:
        # Корреляция с таргетом
        corr_ratio = df[ratio_feat].corr(df['has_anomaly'])
        
        # Находим исходный признак (без _ratio)
        base_feat = ratio_feat.replace('_ratio', '')
        if base_feat in df.columns:
            corr_base = df[base_feat].corr(df['has_anomaly'])
            print(f"\n{ratio_feat}:")
            print(f"  Нормированная версия корреляция: {corr_ratio}")
            print(f"  Исходная версия корреляция: {corr_base}")
            if abs(corr_ratio) > abs(corr_base):
                print(f"  Перфекто! Нормировка улучшила корреляцию на {abs(corr_ratio - corr_base)}")
            else:
                print(f"  АХТУНГ!!! Нормировка не улучшила корреляцию")

print("\nВРЕМЕННОЙ АНАЛИЗ АНОМАЛИЙ:")

df['day'] = pd.to_datetime(df['day'])
df['weekday'] = df['day'].dt.weekday
weekday_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']

anomaly_by_weekday = df[df['has_anomaly'] == 1].groupby('weekday').size()
total_by_weekday = df.groupby('weekday').size()
anomaly_rate_by_weekday = (anomaly_by_weekday / total_by_weekday * 100).fillna(0)

print("Аномалии по дням недели (% от всех действий в этот день):")
for i, name in enumerate(weekday_names):
    print(f"  {name}: {anomaly_rate_by_weekday.get(i, 0)}%")

# График
plt.figure(figsize=(10, 5))
plt.bar(weekday_names, [anomaly_rate_by_weekday.get(i, 0) for i in range(7)], color='salmon')
plt.xlabel('День недели')
plt.ylabel('Доля аномалий (%)')
plt.title('Аномалии по дням недели')
plt.tight_layout()
plt.savefig('plots/anomaly_by_weekday.png', dpi=150)
plt.close()
print("График сохранён: plots/anomaly_by_weekday.png")

# По месяцам
df['month'] = df['day'].dt.month
anomaly_by_month = df[df['has_anomaly'] == 1].groupby('month').size()
total_by_month = df.groupby('month').size()
anomaly_rate_by_month = (anomaly_by_month / total_by_month * 100).fillna(0)

month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']

print("\nАномалии по месяцам (% от всех действий в этом месяце):")
for i, name in enumerate(month_names, 1):
    print(f"  {name}: {anomaly_rate_by_month.get(i, 0)}%")

plt.figure(figsize=(12, 5))
plt.bar(month_names, [anomaly_rate_by_month.get(i, 0) for i in range(1, 13)], color='lightblue')
plt.xlabel('Месяц')
plt.ylabel('Доля аномалий (%)')
plt.title('Аномалии по месяцам')
plt.tight_layout()
plt.savefig('plots/anomaly_by_month.png', dpi=150)
plt.close()
print("График сохранён: plots/anomaly_by_month.png")


print("\nПРОВЕРКА НА ВЫСОКУЮ КОРРЕЛЯЦИЮ МЕЖДУ ПРИЗНАКАМИ:")

# Выбираем числовые признаки (исключая таргет)
numeric_for_corr = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_for_corr = [col for col in numeric_for_corr if col != 'has_anomaly']

corr_matrix_full = df[numeric_for_corr].corr()

# Находим пары с высокой корреляцией
high_corr_pairs = []
for i in range(len(corr_matrix_full.columns)):
    for j in range(i+1, len(corr_matrix_full.columns)):
        if abs(corr_matrix_full.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'feature1': corr_matrix_full.columns[i],
                'feature2': corr_matrix_full.columns[j],
                'correlation': corr_matrix_full.iloc[i, j]
            })

if high_corr_pairs:
    print("Найдены сильно коррелирующие пары признаков (больше 0.8):")
    for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True):
        print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']}")
else:
    print("Сильно коррелирующих пар не найдено — мультиколлинеарности нет")



print("\nПРОВЕРКА ДИСБАЛАНСА ВНУТРИ АНОМАЛИЙ:")

# Сколько аномальных действий в среднем на аномальный день
df['total_bad'] = (df['logon_bad'] + df['device_bad'] + df['http_bad'] + 
                   df['email_bad'] + df['file_bad'])

anomaly_days = df[df['has_anomaly'] == 1]
print(f"Аномальных дней: {len(anomaly_days)}")
print(f"Среднее количество плохих действий в аномальный день: {anomaly_days['total_bad'].mean()}")
print(f"Медиана: {anomaly_days['total_bad'].median()}")
print(f"Максимум: {anomaly_days['total_bad'].max()}")

# Распределение: сколько аномалий из 1 действия, из 2 и т.д.
print("\nРаспределение аномальных дней по количеству плохих действий:")
for i in range(1, 6):
    count = (anomaly_days['total_bad'] == i).sum()
    if count > 0:
        print(f"  {i} плохое действие: {count} дней ({count/len(anomaly_days)*100}%)")


print(f"Все графики сохранены в папке 'plots/'")