import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


def load_data(file_name: str):
    ''' Функция открывает файл с данными.
    Загружает CSV файл из папки data и возвращает его как DataFrame.

    Args:
        file_name (str): Имя файла, например, "train.csv" или "test.csv".

    Returns:
        pd.DataFrame: Данные в виде DataFrame.
    '''

    # Определяем путь к файлу относительно текущего файла
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Папка текущего файла
    data_dir = os.path.join(base_dir, "../data")  # Путь к папке data
    file_path = os.path.join(data_dir, file_name)  # Полный путь к файлу

    try:
        # Загружаем CSV файл в DataFrame
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_name}' не найден в папке '{data_dir}'.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Ошибка: Файл '{file_name}' пуст.")
        return None
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")
        return None


def sweetness_classifier_by_residual_sugar(sugar):
    '''Функция для создания нового признака по ключевому показателю sugar (уровень сахара).
    Используется в процессе предобработки данных внутри функции preproces_data(). '''

    if sugar <= 12:
        return 'dry'
    elif 12 < sugar <= 24:
        return 'off_dry'
    elif 24 < sugar < 50:
        return 'semi_sweet'


def sweetness_classifier_by_density(density):
    '''Функция для создания нового признака по ключевому показателю density (плотность).
    Используется в процессе предобработки данных внутри функции preproces_data(). '''

    if density <= 0.997:
        return 'dry'
    elif 0.997 < density:
        return 'off_dry'


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


def custom_name(input_feature, category):
    '''Функция для создания названия нового признака.
    Используется в OneHotEncoder '''
    return str(input_feature) + "_" + str(category)


def preprocess_data(train_data):
    '''Предварительная подготовка данных: исследование зависимостей, построение новых признаков, удаление ненужных.
    Не могу никак корректно назвать этот этап, потому что формально это не совсем этап предобработки.
    Я бы сказал, отсечение больших кусков, приведение к желаемой форме.'''

    # создание признака сладость по сахару sweetness(residual sugar)
    train_data.insert(4, 'sweetness(residual sugar)', 'Nan')
    train_data['sweetness(residual sugar)'] = train_data['residual sugar'].apply(sweetness_classifier_by_residual_sugar)

    # создание признака сладость по плотности sweetness(density)
    train_data.insert(5, 'sweetness(density)', 'Nan')
    train_data['sweetness(density)'] = train_data['density'].apply(sweetness_classifier_by_density)

    # Создание признака совпадение оценки сладости вина по сахару и по плотности
    train_data['sweetness_accuracy'] = train_data['sweetness(density)'] == train_data['sweetness(residual sugar)']

    # Приведение булевых значений к целочисленному типу (бинарная классификация)
    train_data['sweetness_accuracy'] = train_data['sweetness_accuracy'].astype(int)
    '''
    В итоге мы почти идеально определили сладость согласно двум признакам плотности и остаточного сахар. Их значения очень похожи
    Значения совпадают на 93%. В итоге мы создадим колонку сладость, которую будем считать по колонке остаточный сахар, 
    так как мне кажется что по сахару сладость более корректная. Вопрос - зачем мы настраивали сладость по плотности тогда,
    Если будем брать сладость по сахару. Ответ - мы пытались конечно по обоим признакам настроить некую общую сладость, исходя
    из двух признаков. Настроив некое общее состояние по обоим параметрам, мы получили такую классификацию, в которой они 
    идентичны на 93%. Дальше либо вывести какую то общую логику из двух значений, которая так или иначе будет выбирать показатель
    одного из признаков, либо взять просто сладость по одному из признаков. Я выбираю второй вариант.
    '''

    # Будем брать сладость по колонке остаточного сахара
    train_data.drop(['sweetness_accuracy', 'sweetness(density)'], axis=1, inplace=True)
    train_data.rename(columns={'sweetness(residual sugar)': 'sweetness'}, inplace=True)

    # Удалим выбросы. Два сладких вина - все корректно, но у них многие признаки завышены.
    # train_data.drop([1183, 2738], inplace=True) плохо написал, и когда применял к тестовым данным - сломалось, потому
    # что нет проверки наличия этих индексов
    # Удалить все строки значение оксида в которых больше границы
    train_data.drop(train_data[(train_data['fixed acidity'] > 12)].index, inplace=True)

    # Вставили новый признак wine_body по индексу 6, заполнили пропусками
    train_data.insert(6, 'wine_body', 'Nan')
    # Используем функцию body_classifier(), которая построчно будет вычислять показатель тельности вина
    train_data['wine_body'] = train_data.apply(body_classifier, axis=1)

    # Удалим два признака, которые имеют высокую корреляцию с другими. residual_sugar, density
    # так как из них (вместо них) мы вывели два других: wine_body, sweetness
    train_data.drop(['residual sugar', 'density'], axis=1, inplace=True)
    train_data = train_data.dropna()

    return train_data


def encode_features(X_train, X_valid, X_test):
    """
    Кодирует категориальные признаки на основе тренировочного набора.
    Закодировались: sweetness, wine_body

    Args:
        X_train (pd.DataFrame): Тренировочный набор.
        X_valid (pd.DataFrame): Валидационный набор.
        X_test (pd.DataFrame): Тестовый набор.

    Returns:
        tuple: Закодированные X_train, X_valid, X_test.
    """

    categorical_columns = X_train.select_dtypes(include=["object"]).columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # Кодируем только на тренировочном наборе
    X_train_encoded = pd.DataFrame(
        encoder.fit_transform(X_train[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    X_valid_encoded = pd.DataFrame(
        encoder.transform(X_valid[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    X_test_encoded = pd.DataFrame(
        encoder.transform(X_test[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns)
    )

    # Убираем оригинальные колонки и добавляем закодированные
    X_train = X_train.drop(columns=categorical_columns).reset_index(drop=True)
    X_valid = X_valid.drop(columns=categorical_columns).reset_index(drop=True)
    X_test = X_test.drop(columns=categorical_columns).reset_index(drop=True)

    X_train_onehot = pd.concat([X_train, X_train_encoded], axis=1)
    X_valid_onehot = pd.concat([X_valid, X_valid_encoded], axis=1)
    X_test_onehot = pd.concat([X_test, X_test_encoded], axis=1)

    return X_train_onehot, X_valid_onehot, X_test_onehot


def split_data(data, target_column):
    """
    Разделяет данные на тренировочное и валидационное множества.

    Args:
        data (pd.DataFrame): Исходные данные.
        target_column (str): Имя целевой переменной.

    Returns:
        tuple: X_train, X_valid, y_train, y_valid
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
