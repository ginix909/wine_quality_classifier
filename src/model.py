from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, precision_score
import pandas as pd
from sklearn.model_selection import ShuffleSplit, cross_val_score

def random_forest_tuning(trial, X_train, y_train):
    '''Функция создает поле гипер-параметров модели, для подстановки их на одной итерации. Trial - это одна итерация.
    Возвращает среднее значение параметров перекрестной проверки (cross-validated score) из модели RandomForestClassifier
    после всех итераций (trials). Итерации проходят вне функции, в объекте study.
    Данная функция жестко структурирована в таком исполнении. Требование состоит в том, что функция должна принимать
    только параметр Trial, получая также необходимые параметры X_train,y_train как глобальные переменные.
    То есть функция может работать только непосредственно в окружении этих переменных. Но так как мы дофига порядочные
    люди, теперь мы убрали ее в отдельный файл data_preprocessing.py. По этому, я уберу из функции цикл которые вызывает
    все итерации, и оставлю логику для одной итерации, а цикл на все итерации и поиск результируеющей метрики будет
    вызван непосредственно там, где есть необходимые параметры тренировочного набора в окружении. Функция для одной
     итерации уже может получать тренировочный набор в качестве аргумента, и это не противоречит подразумеваемой
     логике итераций поиска параметров optuna. Так я понял. Если где ошибся, извините, я еще junior). '''

    # Гиперпараметры, которые мы будем настраивать. Я бы сказал - поле гиперпараметров
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 4, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 4, 30)

    # Создаём классификатор с подобранными параметрами
    classifier = RandomForestClassifier(
        criterion=criterion,
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=1
    )

    # Используем кросс-валидацию для оценки модели
    # ShuffleSplit это стратегия разбиения данных дополнительная. Обычная кросс-валидация разибивает просто на 5 частей.
    # А шафл помогает каждый раз еще разбивать по разному.
    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    # кросс-валидация разбивает данные на несколько тренировочных и тестовых множеств (на основе стратегии ShuffleSplit).
    score = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=ss)

    return score.mean()


def evaluate_model(model, X_valid, y_valid, aver):
    '''Принимает на вход обученную модель, валидационный набор, и сравнивает предсказания, которые модель может
    сделать на оценочном наборе и реальный целевой признак оценочного набора. '''
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average=aver)
    # Пришлось установить параметр zero_division =0.0, так как по некоторым классам precision=0,
    # дефолтно метод выбрасывает предупреждение, а так - не выбрасывает. Вообще нулей быть не должно, но там
    # просто нет записей, которые то ли я предсказал бы как класс 3, то ли нет образцов этого класса в y_true.
    recall = recall_score(y_valid, y_pred, average=aver)
    precision = precision_score(y_valid, y_pred, average=aver, zero_division=0.0)

    return accuracy, f1, precision, recall


def make_predictions(model, X_test):
    """
    Выполняет предсказания на тестовом наборе.

    Args:
        model: Обученная модель.
        X_test (pd.DataFrame): Тестовый набор.

    Returns:
        pd.DataFrame: Предсказания.
    """
    predictions = model.predict(X_test)
    return pd.DataFrame(predictions, columns=["quality"])
