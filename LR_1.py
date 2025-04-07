import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from sklearn.impute import SimpleImputer


df = pd.read_csv('autos.csv')

print("\na) head() - первые 5 строк:")
print(df.head())  # Табличный вид

print("\nb) tail() - последние 5 строк:")
print(df.tail())  # Табличный вид


print("\nc) info() - информация о DataFrame:")
print(df.info())  # Табличный вид

print("\nd) describe() - описательная статистика:")
print(df.describe(include='all'))  # Табличный вид

print("\ne) shape - размерность DataFrame:")
print(f"Форма DataFrame: {df.shape}")


# a) Количество пустых значений по колонкам (по убыванию)
null_counts = df.isnull().sum().sort_values(ascending=False)
print("\na) Количество пустых значений по колонкам:")
print(null_counts)  # Табличный вид

# b) Процент пустых значений по колонкам (по убыванию)
null_percent = (df.isnull().mean() * 100).sort_values(ascending=False)
print("\nb) Процент пустых значений по колонкам:")
print(null_percent.round(2))  # Табличный вид

# c) Вывод info до очистки
print("\nc1) Info ДО очистки:")
df.info()

# Очистка данных (удаление строк с пустыми значениями)
df_cleaned = df.dropna()

# Вывод info после очистки
print("\nc2) Info ПОСЛЕ очистки:")
df_cleaned.info()


# Количество строк до удаления дубликатов
rows_before = df.shape[0]
print(f"Количество строк до удаления дубликатов: {rows_before}")

# Находим дубликаты
duplicates = df.duplicated()
num_duplicates = duplicates.sum()
print(f"Найдено дубликатов: {num_duplicates}")

# Удаляем дубликаты
df_no_duplicates = df.drop_duplicates()

# Количество строк после удаления дубликатов
rows_after = df_no_duplicates.shape[0]
print(f"Количество строк после удаления дубликатов: {rows_after}")


# Задания по вариантам


# 1. Заполнение NaN в строковых полях
string_cols = df.select_dtypes(include='object').columns
print(f"Строковые колонки для обработки: {list(string_cols)}")

# Подсчет пропусков до заполнения
print("\nКоличество пропусков до заполнения:")
print(df[string_cols].isnull().sum())

#df = df.fillna('неизвестный')

# импутация для нескольких столбцов
def simple_fast_impute(df, columns):
    """Быстрая импутация медианой (альтернатива KNN)"""
    imputer = SimpleImputer(strategy='median')
    df[columns] = imputer.fit_transform(df[columns].replace(0, np.nan))
    return df

df = simple_fast_impute(df, ['price', 'powerPS'])

# Проверка после заполнения
print("\nРезультат после заполнения пропусков:")
print(df[string_cols].isnull().sum())

# Вывод примера заполненных данных
print("\nПример данных после заполнения:")
print(df[string_cols].head())

# 2. Оценка диапазонов числовых значений
# Выбираем числовые колонки для анализа
numeric_cols = ['yearOfRegistration', 'price', 'powerPS']


# Boxplot для всех числовых колонок сразу
sns.boxplot(data=df[numeric_cols], palette=['skyblue', 'lightgreen', 'salmon'])
plt.title('Распределение числовых показателей (boxplot)')
plt.xticks(ticks=[0, 1, 2], 
           labels=['Год регистрации', 'Цена', 'Мощность'])
plt.ylabel('Значения')
plt.show()


# 3. Удаление выбросов
print(f"Размер данных до удаления выбросов: {df.shape}")
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]
print(f"Размер данных после удаления выбросов: {df.shape}")

# Устанавливаем верхнюю границу для цены
price_upper_bound = 13600

# Фильтруем данные
start_count = df.shape[0]
df = df[df['price'] <= price_upper_bound] 
end_count = df.shape[0]
deleted_count = start_count - end_count

# Статистика после фильтрации
print(f"Статистика фильтрации по цене:")
print(f"Исходное количество записей: {start_count:,}")  
print(f"Оставшееся количество записей: {end_count:,}")
print(f"Удалено записей: {deleted_count:,} ({deleted_count/start_count:.1%})")
print(f"Максимальная цена после фильтрации: {df['price'].max():,.2f} руб.")  


# ПОСЛЕ удаления выбросов
df.boxplot(column=['yearOfRegistration', 'price', 'powerPS'])
plt.title('Диапазоны значений для числовых колонок')
plt.ylabel('Значения')
plt.xlabel('Колонка')
plt.show()



# 4. Матрица корреляций
# Выбираем нужные числовые столбцы
numeric_cols = ['price', 'yearOfRegistration', 'powerPS', 'kilometer', 'monthOfRegistration', 'postalCode']
corr_matrix = df[numeric_cols].corr()

# Создаем таблицу корреляции
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            center=0,
            linewidths=0.5,
            annot_kws={"size": 12})

plt.title('Матрица корреляции числовых признаков', pad=20, fontsize=15)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()


# 5. Удаление малоинформативных столбцов
df.info()
cols_to_drop = ['monthOfRegistration', 'postalCode']
df = df.drop(cols_to_drop, axis=1)
df.info()

# 6. Гистограммы по категориям
plt.figure(figsize=(15, 12))

# Топ 15 марок автомобилей
plt.subplot(3, 1, 1)
brand_counts = df['brand'].value_counts().head(15)
brand_plot = brand_counts.plot(kind='bar', color='skyblue')
plt.title('Топ 15 марок автомобилей', fontsize=14, pad=20)
plt.xlabel('Марка', fontsize=12)
plt.ylabel('Количество', fontsize=12)

# Добавляем подписи значений
for i, v in enumerate(brand_counts):
    brand_plot.text(i, v + 0.5, str(v), ha='center', fontsize=10)

# Типы кузова
plt.subplot(3, 1, 2)
type_counts = df['vehicleType'].value_counts()
type_plot = type_counts.plot(kind='bar', color='lightgreen')
plt.title('Типы кузова', fontsize=14, pad=20)
plt.xlabel('Тип кузова', fontsize=12)
plt.ylabel('Количество', fontsize=12)

# Добавляем подписи значений
for i, v in enumerate(type_counts):
    type_plot.text(i, v + 0.5, str(v), ha='center', fontsize=10)

# Типы топлива
plt.subplot(3, 1, 3)
fuel_counts = df['fuelType'].value_counts()
fuel_plot = fuel_counts.plot(kind='bar', color='salmon')
plt.title('Типы топлива', fontsize=14, pad=20)
plt.xlabel('Тип топлива', fontsize=12)
plt.ylabel('Количество', fontsize=12)

# Добавляем подписи значений
for i, v in enumerate(fuel_counts):
    fuel_plot.text(i, v + 0.5, str(v), ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# 7. Добавление нового признака
df['name_length'] = df['name'].str.len()
print("Добавлен новый признак - длина названия автомобиля")
df.info()
df.head()

# 8. Корреляция цена-мощность
pearson_corr, pearson_p = stats.pearsonr(df['price'], df['powerPS'])
spearman_corr, spearman_p = stats.spearmanr(df['price'], df['powerPS'])

print(f"Коэффициент корреляции Пирсона: {pearson_corr:.3f}, p-value: {pearson_p:.3f}")
print(f"Коэффициент корреляции Спирмена: {spearman_corr:.3f}, p-value: {spearman_p:.3f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.kdeplot(df['price'], fill=True)
plt.title('Распределение цены')

plt.subplot(1, 2, 2)
sns.kdeplot(df['powerPS'], fill=True)
plt.title('Распределение мощности')
plt.tight_layout()
plt.show()