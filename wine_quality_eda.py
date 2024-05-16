'''
Это решение написано для участия в соревновании на платформе Aicrowd
https://www.aicrowd.com/challenges/wineq/dataset_files
Задача - предсказать качество вина исходя из его химического состава по десяти-бальной шкале
Тип целевой переменной - количественная дискретная
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from matplotlib.widgets import Slider
from sklearn.dummy import DummyClassifier

import ML.ml_intro.target_var_types
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, f1_score, accuracy_score, \
    classification_report
import optuna
from optuna.samplers import TPESampler
from sklearn.tree import DecisionTreeClassifier
from optuna.visualization import plot_optimization_history
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore")

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# target_type_indicator(target_type= 'Количественная дискретная')

ML.ml_intro.target_var_types.target_type_indicator(target_type='Количественная дискретная')

original_data = pd.read_csv('wine_quality_train.csv')
print('1. EDA')
print(f'Размерность фрейма: {original_data.shape}')
print(f'Записей: {original_data.shape[0]}, свободных признаков: {original_data.shape[1] - 1}')


# Изначально представлены только численные признаки. Причем все признаки непрерывные.
print('Все свободные признаки - непрерывные численные')
df = original_data.copy()

# 1 EDA
# Исследуем пропущенные значения
missing = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
missing.columns = ["count_gaps"]  # колонку в которой посчитали пропуски назовем count
missing = missing.loc[(missing != 0).any(axis=1)]
print()
print('Проверка на пропуски показала что в тренировочном наборе нет пропущенных значений')

'''
Для каждого признака как var и и для каждого графика как sublot
рисуем график scatterplot (график рассеивания) где по оси х-название признака, данные это весь фрейм,  
'''
# fig, axes = plt.subplots(3, 4, figsize=(16, 7))
# fig.subplots_adjust(left=0.5, bottom=0.5, right=1, top=1, hspace=0.2, wspace=0.2)
#
# plt.tight_layout()
# for var, subplot in zip(list(df.columns), axes.flatten()):
#     sns.scatterplot(x=var, y='quality', data=df, ax=subplot, hue='quality')
#
# plt.show()


'''Построю тепловую диаграмму, чтобы посмотреть корреляцию предикторов между собой и с целевой переменной'''
# sns.set(font_scale=0.6)
# correlation_train = df.corr()
# mask = np.triu(correlation_train.corr())
# plt.figure(figsize=(14, 8))
# sns.heatmap(correlation_train, annot=True, fmt='.2f',cmap='coolwarm',mask=mask, linewidths=1,cbar=False)
# plt.show()
'''
По тепловой диаграмме я вижу:
residual_sugar (остаточный сахар) сильно коррелирует с целевой переменой 0.84
Свободный диоксид ощутимо коррелирует с общим количеством диоксида серы 0.61
Общее количество диоксида серы значимо коррелирует с плотностью 0.53
Сильная обратная корреляция наблюдается между плотностью и алкоголем -0.78

Есть признаки мультиколлинеарности!
Можно еще вычислить VIF (вариационный фактор инфляции)
'''


def calculate_vif(data_frame):
    features = data_frame.columns
    vif_data = pd.DataFrame(columns=['Variable', 'VIF'])

    for i, feature in enumerate(features):
        X = data_frame.drop(feature, axis=1)
        y = data_frame[feature]

        model = sm.OLS(y, sm.add_constant(X)).fit()
        vif = 1 / (1 - model.rsquared)

        vif_data = vif_data.append({'Variable': feature, 'VIF': vif}, ignore_index=True)

    return vif_data


vif_result = calculate_vif(df)
print()
print('VIF/коэффициент инфляции дисперсии на первоначальных данных')
print('Есть признаки с показателем инфляции дисперсии более 10')

# Введем новые переменные вместо старых, для того чтобы избавиться от мультиколинераности но оставить информацию

# 1- й вариант с точностью 84%
# def sweetness_marker_by_residual_sugar(sugar):
#     '''Функция принимает значение показателя residual sugar (остаточный сахар) и возвращает категорию
#     данного вина согласно остаточному сахару. Важно отметить что данные цифры дала CHAT GPT. Однако в статье
#     например для сухого до 10, для сладкого от 35 мг/л.'''
#     if sugar <= 9.5:
#         return 'dry'
#     elif 9.5 < sugar <= 24:
#         return 'semi_dry'
#     elif 24 < sugar < 50:
#         return 'semi_sweet'
#     elif sugar >= 50:
#         return 'sweet'
#
# def sweetness_marker_by_density(density):
#     '''Функция принимает значение показателя density (плотность) и возвращает категорию
#     данного вина согласно плотности. Важно отметить что данные цифры дала CHAT GPT.'''
#     # if density <= 1.005:
#     if density <= 0.998:
#         return 'dry'
#     # elif 1.005 < density < 1.015:
#     elif 0.998 < density < 1.010:
#         return 'semi_dry'
#     elif 1.010 <= density < 1.030:
#         return 'semi_sweet'
#     elif density >= 1.030:
#         return 'sweet'
# print('Сначала получилось 84%, но я попробую настроить до 90%')


# 2-й вариант. пытаюсь настроить до 90%
# Основная проблема что по сахару полусухое, то по плотности сухое
def sweetness_classifier_by_residual_sugar(sugar):
    if sugar <= 12:
        return 'dry'
    elif 12 < sugar <= 24:
        return 'off_dry'
    elif 24 < sugar < 50:
        return 'semi_sweet'
    # elif sugar >= 50:
    #     return 'sweet'


def sweetness_classifier_by_density(density):
    if density <= 0.997:
        return 'dry'
    elif 0.997 < density:
        return 'off_dry'
    # elif 0.997 < density < 1.010: все что ниже это изначальная рабочая версия, но наборов получилось 0 и 1 в категориях
    #     return 'off_dry'          сладкое и полусладкое
    # elif 1.010 <= density < 1.030:
    #     return 'semi_sweet'
    # elif density >= 1.030:
    #     return 'sweet'



df.insert(4, 'sweetness(residual sugar)', 'Nan')
df['sweetness(residual sugar)'] = df['residual sugar'].apply(sweetness_classifier_by_residual_sugar)
df['sweetness(density)'] = df['density'].apply(sweetness_classifier_by_density)

# print(df[['residual sugar', 'sweetness(residual sugar)', 'density', 'sweetness(density)']].head(600))
# print(df.loc[df['sweetness(residual sugar)'] == 'semi_sweet',['residual sugar', 'sweetness(residual sugar)', 'density', 'sweetness(density)']])
# print(df.loc[df['sweetness(density)'] == 'off_dry',['residual sugar', 'sweetness(residual sugar)', 'density', 'sweetness(density)']])
# print(df.loc[df['sweetness(density)'] == 'semi_sweet',['residual sugar', 'sweetness(residual sugar)', 'density', 'sweetness(density)']])
# print(df.loc[df['sweetness(density)'] == 'sweet',['residual sugar', 'sweetness(residual sugar)', 'density', 'sweetness(density)']])


df['sweetness_accuracy'] = df['sweetness(density)'] == df['sweetness(residual sugar)']
df['sweetness_accuracy'] = df['sweetness_accuracy'].astype(int)
sweetness_accuracy = df['sweetness_accuracy'].sum()
print()
print(
    f'Совпадение полученной категории сладости на основе остаточного сахара и  на основе плотности:'
    f'{round(sweetness_accuracy)} из {df.shape[0]}')
print(
    f'Процент совпадений полученной категории сладости по остаточному сахару и плотности:{round(sweetness_accuracy / df.shape[0], 2) * 100}%')
# print('получилось 93%')

# посмотрим где у нас несовпадения
# print(df.loc[df['sweetness_accuracy'] == 0,['residual sugar', 'sweetness(residual sugar)', 'density', 'sweetness(density)', 'sweetness_accuracy']])

'''
В итоге мы почти идеально определили сладость согласно двум признакам плотности и остаточного сахар. Их значения очень похожи
Значения совпадают на 93%. В итоге мы создадим колонку сладость, которую будем считать по колонке остаточный сахар, 
так как мне кажется что по сахару сладость более корректная. Вопрос - зачем мы настраивали сладость по плотности тогда,
Если будем брать сладость по сахару. Ответ - мы пытались конечно по обоим признакам настроить некую общую сладость, исходя
из двух признаков. Настроив некое общее состояние по обоим параметрам, мы получили такую классификацию, в которой они 
идентичны на 93%. Дальше либо вывести какую то общую логику из двух значений, которая так или иначе будет выбирать показатель
одного из признаков, либо взять просто сладость по одному из признаков. Я выбираю второй вариант.
'''
print('Будем брать сладость по колонке остаточного сахара')
df.drop(['sweetness_accuracy', 'sweetness(density)'], axis=1, inplace=True)
df.rename(columns={'sweetness(residual sugar)': 'sweetness'}, inplace=True)

'''
Посмотри на распределение численных признаков, могут быть выбросы. Или как с сахаром, ты обрабатываешь сахар более 50
в цикле, а у тебя вообще макисмальное 31 и одно всего значение на 65.
Значит ли это, что не нужно обрабатывать в цикле сладкие вина, конечно не значит. Есть сладкие вина, значит надо
на них закладываться'''

# Смотрим на распределение, потом доделываем классификацию тельности.
# Удалим выбросы. Два сладких вина - все корректно, но у них многие признаки завышены.
df.drop([1183, 2738], inplace=True)
# Удалить все строки значение оксида в которых больше границы
df = df.drop(df[(df['fixed acidity'] > 12)].index)


# fig, ax = plt.subplots(figsize=(10,8))
# feature = 'alcohol'
# ax.scatter(df[feature], df['quality'])
# ax.set_xlabel(feature)
# ax.set_ylabel('Качество вина')
# plt.show()


# Выведи показатель тела вина
def body_classifier(row):
    '''Функция принимает на входе строку фрейма и использует признаки: содержание алкоголя, экстракты (кислотность,
    остаточный сахар) - и определяет на их основе тип тельности вина: Легкое вино, среднее вино, крепкое вино.
    В классическом подходе мы также должны отталкиваться от сорта винограда, на крайний случай от страны происхождения
    Вино из Калифорнии, Италии, Австралии и других солнечных теплых стран всегда будут более полнотелыми в отличие от
    вин из Германии или Франции.
    Так как сахара у полнотелого вина по найденной мной классификации варьируется от 0 до 15 то есть включает ровно тот
    жен коридор что и в условиях для легкого и среднего вина, значит для него сахара вообще не важен в нашем случае. '''
    # это немного лишняя надстройка, но мне так проще будет настраивать границы категорий
    alc = row['alcohol']
    pH = row['pH']
    RS = row['residual sugar']
    # if alc < 12 and pH < 3.2 and RS <= 5:
    # if alc < 12 and pH < 3.4 and RS < 9:
    # if alc <= 12.5 and pH < 3.4 and RS < 11:       Лучший вариант по всем правилам
    if alc <= 11.5 and pH < 3.4 and RS < 11:         #вариант для уменьшения группы легких вин, чтобы увеличить остальные
        return 'light'
    # elif 12 <= alc <= 13.5 and 3.2 <= pH <= 3.5 and RS <= 10:  первоначальный вариант по классификации из теории
    # elif 8 <= alc <= 13 and pH <= 3.66 and RS <= 20.8:       хороший вариант по правилам
    # elif 8 <= alc <= 13 and pH <= 3.66 and RS <= 23.5:       Лучший вариант по правилас, чтобы впустить все записи в группы
    elif 8 <= alc <= 12 and pH <= 3.66 and RS <= 23.5:         # вариант, чтобы увеличить группу полнотелых за счет уменьш среднетелых
        return 'medium'
    # elif 13.5 < alc and 3.5 < pH and RS <= 15:
    # elif 9.5 <= alc <= 15 and pH <= 3.9 and RS <= 27:        Лучший вариант по всем правилам
    elif 9 <= alc <= 15 and pH <= 3.9 and RS <= 27:
        return 'full-bodied'


# Вставили новый признак wine_body по индексу 6, заполнили пропусками
df.insert(6, 'wine_body', 'Nan')
df['wine_body'] = df.apply(body_classifier, axis=1)

# Проверь по категориям, нормальные ли показатели у вин в каждой категории тельности - все категории меня устраивают
print()
print(f'Процент записей с неопределенной тельностью вина: {df["wine_body"].isna().sum() / df.shape[0] * 100}')
# print(df.loc[df['wine_body'].isnull(), ['residual sugar', 'pH', 'alcohol', 'wine_body', 'sweetness']])


# 1 вариант - Удалим два признака, которые имеют высокую корреляцию с другими. residual_sugar, density
# так как из них (вместо них) мы вывели два других: wine_body, sweetness
df.drop(['residual sugar', 'density'], axis=1, inplace=True)
# Снова посмотрим на показатель VIF/variance inflation factor(коэффициент инфляции дисперсии)
numeric_features = [col for col in df.columns if df[col].dtype ]
numeric_features = df.select_dtypes(include=['int','float']).columns.tolist()
print(numeric_features)
# vif_result = calculate_vif(df[numeric_features])
print('Мультиколлинеарность устранили: 1 вариант - вывести новые признаки вместо текущих с высокой мультиколинеарностью')
# print('VIF/коэффициент инфляции дисперсии после удаления двух признаков с высокими показателями VIF (более 10)')
# print(vif_result)


# 2 вариант - просто удалить два мультиколлинеарных признака residual_sugar, density

# 3 вариант - оставить все признаки несмотря на мультиколлинеарность

print()
print(f'Теперь в дата-фрейме:{df.shape[1]} признаков. '
      f'Численных: {len(df.select_dtypes(include=["int","float"]).columns.tolist())-1}. '
      f'Категориальных: {len(df.select_dtypes(include=["object"]).columns.tolist())}. '
      f'Целевая переменная: 1.')


# 1-й Вариант использовать несбалансированные группы по более правильным критериям отбора

# 2-й Вариант немного изменить группы в угоду лучшей сбалансированности
# Я выбрал второй вариант
'''
# Блок проверки тельности по группам
print('контроль по группам тельности:')
print(f'Легкие вина: {df[df["wine_body"] == "light"].shape[0]} штук')
print(f'Средние вина: {df[df["wine_body"] == "medium"].shape[0]} штук')
print(f'Полнотелые вина: {df[df["wine_body"] == "full-bodied"].shape[0]} штук')
marker = (df[df["wine_body"] == "light"].shape[0] + df[df["wine_body"] == "medium"].shape[0] +
          df[df["wine_body"] == "full-bodied"].shape[0]) == df.shape[0]
print(f'Общее количество соответствует изначальному количеству записей: {marker}')'''
# Легкие вина: 2528 штук   - стало 2054
# Средние вина: 1285 штук  - стало 1270
# Полнотелые вина: 99 штук - стало 591
# Количество изменилось -потому что я подумал, что группы должны быть более представительными



# sns.set(font_scale=0.6)
# correlation_train = df.corr()
# mask = np.triu(correlation_train.corr())
# plt.figure(figsize=(14, 8))
# sns.heatmap(correlation_train, annot=True, fmt='.2f',cmap='coolwarm',mask=mask, linewidths=1,cbar=False)
# plt.show()


'''# Проверка!
# Подумал о проблеме диспропорции категории внутри признака (категориального)
# Не нашел такой проблемы в интернете, но она очевидно существует, по крайней мере для меня
# Посчитаю количество записей внутри каждой из категориальных переменных
print('')
q = df.shape[0]
print('Checking category disproportion of categorical feature')
# print(f'Сладких вин внутри категории sweetness: {df[df["sweetness"] == "sweet"].shape[0]}')
# print(f'Сладких вин доля во фрейме: {df[df["sweetness"] == "sweet"].shape[0]/q}')
# print(f'Полусладких вин внутри категории sweetness: {df[df["sweetness"] == "semi_sweet"].shape[0]}')
# print(f'Полусладких вин доля во фрейме: {round(df[df["sweetness"] == "semi_sweet"].shape[0]/q,2)}')
print(f'Полусухих вин внутри категории sweetness: {df[df["sweetness"] == "off_dry"].shape[0]}')
print(f'Полусухих вин доля во фрейме: {round(df[df["sweetness"] == "off_dry"].shape[0]/q, 2)}')
print(f'Сухих вин внутри категории sweetness: {df[df["sweetness"] == "dry"].shape[0]}')
print(f'Cухих вин доля во фрейме: {round(df[df["sweetness"] == "dry"].shape[0]/q, 2)}')
print('')
print(f'Легких вин внутри категории wine_body: {df[df["wine_body"] == "light"].shape[0]}')
print(f'Легких вин доля во фрейме: {round(df[df["wine_body"] == "light"].shape[0]/q, 2)}')
print(f'Среднетелых вин внутри категории wine_body: {df[df["wine_body"] == "medium"].shape[0]}')
print(f'Среднетелых вин доля во фрейме: {round(df[df["wine_body"] == "medium"].shape[0]/q, 2)}')
print(f'Полнотелых вин внутри категории wine_body: {df[df["wine_body"] == "full-bodied"].shape[0]}')
print(f'Полнотелых вин доля во фрейме: {round(df[df["wine_body"] == "full-bodied"].shape[0]/q, 2)}')
# Ну вот есть какие-то способы учесть дисбаланс - но я не могу из 1 записи раздуть какую то важность!
# Я вижу один способ - не создавать категорию сладкого вина, если по статистике (точнее по моей группировке)
# таких вин получилось 1 штука, то никакой значимости группа не несет
# Вот полнотелых вин должно быть больше!!!!
print('Checking category disproportion of categorical feature')
# По итогу я убрал группу сладких вин (объединил ее с полусладкими), изменил пропорции в группах сладости и тельности 
# представительность малых подгрупп увеличилась. 15 и 17 процентов самые маленькие группы. Более 10% я хотел.
'''

# Пора делить на тренировочное и тестовое множество для того чтобы оценивать качество предсказаний

print(' ')
print('2. Предподготовка данных')
y = df['quality']
df.drop('quality', axis=1, inplace=True)
X_train, X_valid, y_train, y_valid = train_test_split(df, y, random_state=0, train_size=0.8)


print(f'Закодируем категориальные колонки {X_train.select_dtypes(include="object").columns.tolist()}')

def custom_name(input_feature, category):
    return str(input_feature) + "_" + str(category)

# Выделим готовые численные колонки, чтобы потом приклеить к ним закодированные

categorical_cols_train = X_train.select_dtypes(include='object').columns.tolist()
categorical_cols_valid = X_valid.select_dtypes(include='object').columns.tolist()
# print(f'categorical_cols_train: {categorical_cols_train}')
# print(f'categorical_cols_valid: {categorical_cols_valid}')

numeric_cols_train = X_train.drop(categorical_cols_train, axis=1)
numeric_cols_valid = X_valid.drop(categorical_cols_valid, axis=1)

# Закодируем категориальные колонки wine_body, sweetness в тренировочном и оценочном наборах
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, feature_name_combiner= custom_name)
encoded_train = pd.DataFrame(OH_encoder.fit_transform(X_train[categorical_cols_train]))
encoded_valid = pd.DataFrame(OH_encoder.transform(X_valid[categorical_cols_valid]))

# Назначим всем колонкам корректные названия
encoded_train.columns = OH_encoder.get_feature_names_out()
encoded_valid.columns = OH_encoder.get_feature_names_out()

# Назначим всем закодированным колонкам корректные индексы
encoded_train.index = X_train.index
encoded_valid.index = X_valid.index

# Склеим выделенные вначале численные колонки с закодированными колонками
X_train_onehot = numeric_cols_train.join(encoded_train)
X_valid_onehot = numeric_cols_valid.join(encoded_valid)


# Проверим типы данных признаков - float64
# print(onehot_train_df.dtypes)

# Проверим что названия колонок имеет тип данных str - в дальнейшем это может помешать
# Хотя формально я вручную называл новые признаки и клеил строку

def check_col_name_type(data_frame):
    '''Функция определит тип данных НАЗВАНИЯ колонки/признака
    На вход получает дата-фрейм.
    Возвращает массив, которые содержит итеративно накопленные типы данных колонок'''
    column_name_list = data_frame.columns.to_list()
    column_name_type_list = []
    for column_name in column_name_list:
        column_name_type_list.append(type(column_name))
    return column_name_type_list

# print(check_col_name_type(onehot_train_df))

# Проверим не появились ли пропуски после кодирования - было такое, когда я не назначал корректные индексы
# print(onehot_train_df.isnull().sum())
# print(onehot_valid_df.isnull().sum())



'''Хочу посмотреть на корреляцию новых признаков с целевой переменной\
Я никогда пожалуй не смотрел на корреляцию после горячего кодирования. Что тут видно:
Между категориями внутри признака есть прямая обратная корреляция. Сначала я испугался. А по факту:
У каждого сэмпла (запись) по одной изначальной категории есть например три новых признака, и там конечно будет 
именно полная обратная корреляция, так как если у одного нового подпризнака стоит единичка, то у других подпризнаков
будут стоять нули.
Тогда это наверное должно работать между всеми категориями, возможно тут важна пропорциональность категорий.
'''

# sns.set(font_scale=0.6)
# correlation_train = onehot_train_df.corr()
# mask = np.triu(correlation_train.corr())
# plt.figure(figsize=(12, 6))
# sns.heatmap(correlation_train, annot=True, fmt='.2f',cmap='coolwarm',mask=mask, linewidths=1,cbar=False)
# plt.show()


'''И вот здесь вопрос! А был ли смысл добавления новых признаков!!
Изначальная идея была убрать мультиколлинеарность между исходными признаками - прошла между исходными, но появилась
внутри категорий. 
Также итоговый смысл повысить точность предсказаний - посмотрим на предсказаниях.'''



# Сейчас я изучаю вопрос классового дисбаланса, потому что у меня явно данных некоторых классов меньше
# Диспропорции в категориях нужно выявить и устранить перед кодированием! или не нужно???
# Я посчитал что нужно, и увеличил представительность малых категорий, одну удалил (в ней была 1 запись)


# Что будем делать дальше? посмотри в decision way!
# По-хорошему надо собрать препроцессор для деревовидной модели - так код будет более лаконично, читаемо и масштабируемо
# Но я почитал свои пометки: конвейер тяжело настраивать.


# Возьмем несколько моделей и настроим их отдельно
# Настройка модели DecisionTreeClassifier()
def decision_tree_tunnig(trial):
    '''Функция создает поле гипер-параметров модели, для подстановки их на одной итерации. Trial - это одна итерация.
    Возвращает среднее значение параметров перекрестной проверки (cross-validated score) из модели DecisionTreeClassifier
    после всех итераций (trials). Итерации проходят вне функции, в объекте study.'''

    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 4,30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 4,30)

    classifier = DecisionTreeClassifier(max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        criterion=criterion,
                                        random_state=1)

    # перемешаем данные для наилучшего обучения
    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    score = cross_val_score(classifier, X_train_onehot, y_train, scoring='accuracy', cv=ss)
    # score = cross_val_score(classifier, onehot_train_df, y_train, scoring='f1_weighted', cv=ss)
    score = score.mean()
    return score


trials_quantity=20
sampler = TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(decision_tree_tunnig, n_trials=trials_quantity)
# fig = plot_optimization_history(study)
# fig = optuna.visualization.plot_slice(study)
# # fig = plot_contour(study)
# # fig = plot_param_importances(study)
# fig.show()
best_trial = study.best_trial
best_parameters = study.best_params
print(f'Лучшее испытание (trial): №{best_trial.number} за {trials_quantity} итераций модели DecisionTreeClassifier.')
print(f'Лучший показатель accuracy: {best_trial.values[0]} ')
'''
# Думаю нужно взять f1_macro она для многоклассовой классификации, учитывают дисбаланс классов
print(f'Лучший показатель f1_macro: {round(best_trial.values[0],3)} ')
print(f'Лучшие параметры, полученные в ходе {trials_quantity} итерации обучения модели DecisionTreeClassifier: {best_parameters}')
print('Получился низкий показатель F-меры 0.319 Это плохой показатель!!! ')
'''




# Настройка модели DecisionTreeClassifier()

def random_forest_tunnig(trial):
    '''Функция создает поле гипер-параметров модели, для подстановки их на одной итерации. Trial - это одна итерация.
    Возвращает среднее значение параметров перекрестной проверки (cross-validated score) из модели RandomForestClassifier
    после всех итераций (trials). Итерации проходят вне функции, в объекте study.'''

    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    n_estimators = trial.suggest_int('n_estimators', 10,100)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 4,30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 4,30)


    classifier = RandomForestClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf, random_state=1)

    # перемешаем данные для наилучшего обучения
    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    # f1_scorer = make_scorer(f1_score, average='weighted')
    score = cross_val_score(classifier, X_train_onehot, y_train, scoring='accuracy', cv=ss)
    score = score.mean()
    return score


trials_quantity=10
sampler = TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(random_forest_tunnig, n_trials=trials_quantity)
# fig = plot_optimization_history(study)
# fig = optuna.visualization.plot_slice(study)
# fig = plot_contour(study)
# fig = plot_param_importances(study)
# fig.show()
best_trial = study.best_trial
best_parameters = study.best_params
print(f'Лучшее испытание (trial): №{best_trial.number} за {trials_quantity} итераций модели RandomForestClassifier.')
print(f'Лучший показатель accuracy: {round(best_trial.values[0],3)} ')
print(f'Лучшие параметры, полученные в ходе {trials_quantity} итерации обучения модели RandomForestClassifier: {best_parameters}')
# print('f1_weighted это усредненная F1-мера по всем классам.Получилось 0.569. Надо более 0,750 я думаю ')


# Плохие результаты метрик
# Строй еще одну модель посерьезнее
# Потом пробуй другие сценарии предобработки данных

def KMeans_tunning(trial):
    '''Функция создает поле гипер-параметров модели, для подстановки их на одной итерации. Trial - это одна итерация.
    Возвращает среднее значение параметров перекрестной проверки (cross-validated score) из модели KMeans
    после всех итераций (trials). Итерации проходят вне функции, в объекте study.'''

    n_clusters = trial.suggest_int('n_clusters', 1,10)
    init = trial.suggest_categorical('init', ['k-means++', 'random'])
    max_iter = trial.suggest_int('max_iter', 100,300, step=50)
    algorithm = trial.suggest_categorical('algorithm', ['lloyd', 'elkan'])

    classifier = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, algorithm=algorithm, random_state=1)

    # перемешаем данные для наилучшего обучения
    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    # f1_scorer = make_scorer(f1_score, average='weighted')
    score = cross_val_score(classifier, X_train_onehot, y_train, scoring='accuracy', cv=ss)
    score = score.mean()
    return score

trials_kmeans = 40
sampler_kmeans = TPESampler(seed=42)
study_kmeans = optuna.create_study(direction="maximize", sampler=sampler_kmeans)
study_kmeans.optimize(KMeans_tunning, n_trials=trials_kmeans)
# fig = plot_optimization_history(study_kmeans)
# fig = optuna.visualization.plot_slice(study)
# fig = plot_contour(study)
# fig = plot_param_importances(study)
# fig.show()
best_trial_kmeans = study_kmeans.best_trial
best_parameters_kmeans = study_kmeans.best_params
print(f'Лучшее испытание (trial): №{best_trial_kmeans.number} за {trials_kmeans} итераций модели KMeans.')
print(f'Лучший показатель accuracy: {round(best_trial_kmeans.values[0],3)} ')
print(f'Лучшие параметры, полученные в ходе {trials_kmeans} итерации обучения модели KMeans: {best_parameters_kmeans}')
#  я попробовал f1_score, homogeneity_score, Silhouette Score, Davies-Bouldin Index... работает гомогенность, но я не
# понимаю что она отражает

'''
Вопрос такой: какими метриками оценивать модели? 
Для kmeans якобы нельзя использовать класстические метрики. 
Тогда как понять какая модель лучше работает.
Давай ты посмотришь какие модели используют для многоклассовой классификации заново и узнаешь какими метриками их оценивают?
'''



# def ____tunning(trial):


# 4. CatBoost

# Посмотрим на рспределение целевой переменной
plt.figure(figsize=(8, 4))
sns.countplot(x=y_train)
plt.title('Distribution of Target Classes')
plt.xlabel('Class')
plt.ylabel('Count')
# plt.show()

# классы не сбалансированы

# обучим модель
catboost_model = CatBoostClassifier( iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', verbose=False)
catboost_model.fit(X_train_onehot, y_train)

# сделаем предсказания
y_pred = catboost_model.predict(X_valid_onehot)

# Опишем метрику точности и отчет
accuracy = accuracy_score(y_valid, y_pred)
report = classification_report(y_valid, y_pred)
# выведем метрики
print(f"Accuracy модели с фиксированными параметрами: {accuracy}")
print(f"Classification Report. Отчет по модели {catboost_model.__class__.__name__}.")
print(report)


'''Tunning data - попробуем по другому предобработать данные

1 вариант: не вводим признак тело вина wine_body и сладость. 
    Не повысилась точность

2 вариант: вернем параметры, которые сильно коррелируют друг с другом и с некоторыми переменными содержание сахара и плотность
    общие показатели не изменились
    хотя по группам стало лучше: теперь хоть как то предсказываются оценки 4,5,5,6,7,8 (кроме 3)
    не пойдет!

Давай посмотрим какую точность покажет модель, которая рандомно будет распределять вино по оценкам


'''

random_model = DummyClassifier(strategy='stratified', random_state=42)
random_model.fit(X_train_onehot, y_train)

# Получаем предсказания на тестовых данных
y_pred = random_model.predict(X_valid_onehot)

# Оцениваем качество модели
accuracy = accuracy_score(y_valid, y_pred)
classification_rep = classification_report(y_valid, y_pred)

# Выводим результаты
print(f'Accuracy рандомной классификации: {accuracy:.2f}')
print(f'Classification Report. Отчет по модели {random_model.__class__.__name__}')
print(classification_rep)




'''
Какие выводы я имею сейчас
моя лучшая модель пока catboost показывает f-меру среднюю 0.6 это мало
Рандомная классификация показывает f-меру среднюю 0.34. То есть обьясненная вариация примерно 0.26 это мало

Что еще можно сделать:
0. Можно попробовать подобрать лучшие параметры для модели catboost
Можно понадеяться на мета-модель, которая может быть догонит до 0.7 f-меру
Можно посмотреть еще раз на данные.
Больше всего я верю сейчас в более внимательную предподготовку данных.
    1.Обрезать выбросы
    2.удалить возможно крайние категории (не знаю правда делают ли так)
    3. добавить новые признаки - добавлял, подумай что еще можно добавить
    4. Посмотри как модель предскажет по двум твоим фичам чисто, чтобы понять - фигню ты закодил или нет


Из позитивного что могу сказать: все таки какая никакая определенность есть в моих предсказаниях
и мы поднимать будем не с нуля. 
Что ты хотел? чтобы придумал две пусть и гениальные колонки и модель все предскажет в единицу ? 
Конечно нет. Надо поработать. ДА сложно, да непонятно куда идти, но это точно еще не тупик
'''

'''
Подберем лучшие параметры для модели catboost
'''

def catboost_tunning(trial):
    '''Функция создает поле гипер-параметров модели, для подстановки их на одной итерации. Trial - это одна итерация.
    Возвращает среднее значение параметров перекрестной проверки (cross-validated score) из модели CatBoostClassifier
    после всех итераций (trials). Итерации проходят вне функции, в объекте study.'''
    params = {
    'iterations': trial.suggest_int('iterations', 100, 1000),
    'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
    'depth': trial.suggest_int('depth', 4, 10),
    'random_strength': trial.suggest_loguniform('random_strength', 0.01, 1.0),
    'border_count': trial.suggest_int('border_count', 5, 255),
    'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 0.1, 10.0),
    }
    classifier = CatBoostClassifier(**params, random_state=1, verbose=False)

    # перемешаем данные для наилучшего обучения
    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    # f1_scorer = make_scorer(f1_score, average='weighted')
    # score = cross_val_score(classifier, X_train_onehot, y_train, scoring='homogeneity_score', cv=ss)
    score = cross_val_score(classifier, X_train_onehot, y_train, scoring='accuracy', cv=ss)
    score = score.mean()
    return score

'''
trials_catboost = 20
sampler_catboost = TPESampler(seed=42)
study_catboost = optuna.create_study(direction="maximize", sampler=sampler_catboost)
study_catboost.optimize(catboost_tunning, n_trials=trials_catboost)
# fig = plot_optimization_history(study_catboost)
# fig = optuna.visualization.plot_slice(study)
# fig = plot_contour(study)
# fig = plot_param_importances(study)
# fig.show()
best_trial_catboost = study_catboost.best_trial
best_parameters_catboost = study_catboost.best_params
print(f'Лучшее испытание (trial): №{best_trial_catboost.number} за {trials_catboost} итераций модели CatBoostClassifier.')
print(f'Лучший показатель accuracy: {round(best_trial_catboost.values[0],3)} ')
print(f'Лучшие параметры, полученные в ходе {trials_catboost} итерации обучения модели CatBoostClassifier: {best_parameters_catboost}')
print('Лучшую точность показала модель CatBoostClassifier с настроенными гипер-параметрами')
'''


# Напиши еще одну модель catboost передай ей лучшие параметры и убедись что точность такая же
catboost_best = CatBoostClassifier( iterations=791, depth=10, learning_rate=0.008932595944810082,
                                    loss_function='MultiClass',l2_leaf_reg= 0.266232288831554, verbose=False)
catboost_best.fit(X_train_onehot, y_train)

# сделаем предсказания
y_pred = catboost_best.predict(X_valid_onehot)

# Опишем метрику точности и отчет
accuracy = accuracy_score(y_valid, y_pred)
f1 = f1_score(y_valid,y_pred)
report = classification_report(y_valid, y_pred)
# выведем метрики
print(f"Accuracy модели с лучшими фиксированными параметрами: {accuracy}")
print(f"F1 модели с лучшими фиксированными параметрами: {f1}")
print(f"Classification Report. Отчет по модели {catboost_best.__class__.__name__}.")
print(report)



# Выгружай в документ и отправляй в aicrowd

# В конце можешь организовать решение в виде конвейера, так как не нужно уже настраивать, так будет как бы красивее