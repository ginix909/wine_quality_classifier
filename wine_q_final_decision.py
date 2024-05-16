
'''
Это решение написано для участия в соревновании на платформе Aicrowd
https://www.aicrowd.com/challenges/wineq/dataset_files
Задача - предсказать качество вина исходя из его химического состава по десяти-бальной шкале
Тип целевой переменной - количественная дискретная
'''

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

original_data = pd.read_csv('wine_quality_train.csv')
test_data = pd.read_csv('wine_q_test.csv')
# print(f'Размерность фрейма: {original_data.shape}')
# print(f'Записей: {original_data.shape[0]}, свободных признаков: {original_data.shape[1] - 1}')

# Изначально представлены только численные признаки. Причем все признаки непрерывные.
# print('Все свободные признаки - непрерывные численные')
df = original_data.copy()

# 1 EDA
# Исследуем пропущенные значения
missing = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
missing.columns = ["count_gaps"]  # колонку в которой посчитали пропуски назовем count
missing = missing.loc[(missing != 0).any(axis=1)]
# print()
# print('Проверка на пропуски показала что в тренировочном наборе нет пропущенных значений')

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


# Введем новые переменные вместо старых, для того чтобы избавиться от мультиколинераности но оставить информацию

def sweetness_classifier_by_residual_sugar(sugar):
    if sugar <= 12:
        return 'dry'
    elif 12 < sugar <= 24:
        return 'off_dry'
    elif 24 < sugar < 50:
        return 'semi_sweet'


def sweetness_classifier_by_density(density):
    if density <= 0.997:
        return 'dry'
    elif 0.997 < density:
        return 'off_dry'


df.insert(4, 'sweetness(residual sugar)', 'Nan')
df['sweetness(residual sugar)'] = df['residual sugar'].apply(sweetness_classifier_by_residual_sugar)
df['sweetness(density)'] = df['density'].apply(sweetness_classifier_by_density)

df['sweetness_accuracy'] = df['sweetness(density)'] == df['sweetness(residual sugar)']
df['sweetness_accuracy'] = df['sweetness_accuracy'].astype(int)
sweetness_accuracy = df['sweetness_accuracy'].sum()

'''
В итоге мы почти идеально определили сладость согласно двум признакам плотности и остаточного сахар. Их значения очень похожи
Значения совпадают на 93%. В итоге мы создадим колонку сладость, которую будем считать по колонке остаточный сахар, 
так как мне кажется что по сахару сладость более корректная. Вопрос - зачем мы настраивали сладость по плотности тогда,
Если будем брать сладость по сахару. Ответ - мы пытались конечно по обоим признакам настроить некую общую сладость, исходя
из двух признаков. Настроив некое общее состояние по обоим параметрам, мы получили такую классификацию, в которой они 
идентичны на 93%. Дальше либо вывести какую то общую логику из двух значений, которая так или иначе будет выбирать показатель
одного из признаков, либо взять просто сладость по одному из признаков. Я выбираю второй вариант.
'''
# print('Будем брать сладость по колонке остаточного сахара')
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

'''
Попробуем обработать выбросы с помощью нового инструмента - IQR (interquantile range)
этот метод написан в проба_лучших_методов.py
# 1-способ просто удалим значения не глядя на них
q1 = df_wine['fixed acidity'].quantile(0.25)
q3 = df_wine['fixed acidity'].quantile(0.75)
IQR = q3 - q1
low_border = q1 - 1.5 * IQR
high_border = q3 + 1.5 * IQR
Так как выбросы лежат за границами заданного интервала, значит мы оставляем во фрейме значения, которые наоборт ограничены
вычисленным верхним и нижним значением
df_wine = df_wine.loc[(df_wine['fixed acidity'] > low_border) & (df_wine['fixed acidity'] < high_border)]
'''



# Выведи показатель тела вина
def body_classifier(row):
    '''Функция принимает на входе строку фрейма и использует признаки: содержание алкоголя, экстракты (кислотность,
    остаточный сахар) - и определяет на их основе тип тельности вина: Легкое вино, среднее вино, крепкое вино.
    В классическом подходе мы также должны отталкиваться от сорта винограда, на крайний случай от страны происхождения
    Вино из Калифорнии, Италии, Австралии и других солнечных теплых стран всегда будут более полнотелыми в отличие от
    вин из Германии или Франции.
    Так как сахара у полнотелого вина по найденной мной классификации варьируется от 0 до 15 то есть включает ровно тот
    жен коридор что и в условиях для легкого и среднего вина, значит для него сахара вообще не важен в нашем случае. '''
    alc = row['alcohol']
    pH = row['pH']
    RS = row['residual sugar']
    if alc <= 11.5 and pH < 3.4 and RS < 11:  # вариант для уменьшения группы легких вин, чтобы увеличить остальные
        return 'light'
    elif 8 <= alc <= 12 and pH <= 3.66 and RS <= 23.5:  # вариант, чтобы увеличить группу полнотелых за счет уменьш среднетелых
        return 'medium'
    elif 9 <= alc <= 15 and pH <= 3.9 and RS <= 27:
        return 'full-bodied'


# Вставили новый признак wine_body по индексу 6, заполнили пропусками
df.insert(6, 'wine_body', 'Nan')
df['wine_body'] = df.apply(body_classifier, axis=1)



# 1 вариант - Удалим два признака, которые имеют высокую корреляцию с другими. residual_sugar, density
# так как из них (вместо них) мы вывели два других: wine_body, sweetness
df.drop(['residual sugar', 'density'], axis=1, inplace=True)
# Снова посмотрим на показатель VIF/variance inflation factor(коэффициент инфляции дисперсии)
numeric_features = [col for col in df.columns if df[col].dtype]
numeric_features = df.select_dtypes(include=['int', 'float']).columns.tolist()


y = df['quality']
df.drop('quality', axis=1, inplace=True)
X_train, X_valid, y_train, y_valid = train_test_split(df, y, random_state=0, train_size=0.8)


def custom_name(input_feature, category):
    return str(input_feature) + "_" + str(category)


categorical_cols_train = X_train.select_dtypes(include='object').columns.tolist()
categorical_cols_valid = X_valid.select_dtypes(include='object').columns.tolist()

numeric_cols_train = X_train.drop(categorical_cols_train, axis=1)
numeric_cols_valid = X_valid.drop(categorical_cols_valid, axis=1)

# Закодируем категориальные колонки wine_body, sweetness в тренировочном и оценочном наборах
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, feature_name_combiner=custom_name)
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


def check_col_name_type(data_frame):
    '''Функция определит тип данных НАЗВАНИЯ колонки/признака
    На вход получает дата-фрейм.
    Возвращает массив, которые содержит итеративно накопленные типы данных колонок'''
    column_name_list = data_frame.columns.to_list()
    column_name_type_list = []
    for column_name in column_name_list:
        column_name_type_list.append(type(column_name))
    return column_name_type_list


def random_forest_tunnig(trial):
    '''Функция создает поле гипер-параметров модели, для подстановки их на одной итерации. Trial - это одна итерация.
    Возвращает среднее значение параметров перекрестной проверки (cross-validated score) из модели RandomForestClassifier
    после всех итераций (trials). Итерации проходят вне функции, в объекте study.'''

    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 4, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 4, 30)

    classifier = RandomForestClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf, random_state=1)

    # перемешаем данные для наилучшего обучения
    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    score = cross_val_score(classifier, X_train_onehot, y_train, scoring='accuracy', cv=ss)
    score = score.mean()
    return score


# Лучшая модель random forest
random_forest_best = RandomForestClassifier(criterion='entropy', max_depth=15, n_estimators=63,
                                            min_samples_split=4, min_samples_leaf=4, random_state=1)

random_forest_best.fit(X_train_onehot, y_train)
y_pred = random_forest_best.predict(X_valid_onehot)
accuracy = accuracy_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred, average='macro')
report = classification_report(y_valid, y_pred)
print(f"Accuracy модели с лучшими фиксированными параметрами: {accuracy}")
print(f"f1_macro модели с лучшими фиксированными параметрами: {f1}")
# print(f"Classification Report. Отчет по модели {random_forest_best.__class__.__name__}.")


# Test data preprocessing
test_data.insert(4, 'sweetness', 'Nan')
test_data['sweetness'] = test_data['residual sugar'].apply(sweetness_classifier_by_residual_sugar)
test_data.insert(6, 'wine_body', 'Nan')
test_data['wine_body'] = test_data.apply(body_classifier, axis=1)
test_data.drop(['residual sugar', 'density'], axis=1, inplace=True)

categorical_cols_train = test_data.select_dtypes(include='object').columns.tolist()
numeric_cols_train = test_data.drop(categorical_cols_train, axis=1)

# Закодируем категориальные колонки (wine_body, sweetness) в тестовом
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, feature_name_combiner=custom_name)
encoded_train = pd.DataFrame(OH_encoder.fit_transform(test_data[categorical_cols_train]))
encoded_train.columns = OH_encoder.get_feature_names_out()
encoded_train.index = test_data.index
test_onehot = numeric_cols_train.join(encoded_train)
test_onehot.drop('wine_body_None', axis=1, inplace=True)

# Делаем предсказания
submission = random_forest_best.predict(test_onehot)
submission = pd.DataFrame(submission)
submission.to_csv('submission.csv', header=['quality'], index=False)


# Я не выложил решение, потому что соревнование закончилось, и получается уже нельзя загрузить решение