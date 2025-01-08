import sys
import os
import pytest

import optuna
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

from ML.Aicrowd.wine_q.src.model import random_forest_tuning, evaluate_model

# Добавляем корневую папку проекта в пути поиска
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


def test_random_forest_tuning():
    """
    Тестирует функцию random_forest_tuning:
    1. Проверяет корректную работу с фиктивными данными. Создал фиктивный дата-сет, разделили на тренировочное и
    оценочное множество, создали объект полного обучения study, вызывли у него метод оптимизации, которому передали
    мою функцию, обернутую в самодельную тривиальную функцию objective(). Можно было бы передать просто мою функцию, но
    функция должна принимать только объект trial, а моя функция принимает еще X_train, y_train. По этому такая надстройка.
    Я это больше для себя пишу, чтобы лучше понимать.

    2. Убеждается, что возвращается значение метрики (accuracy).
    Получили не пустое значение best_value
    Значение best_value является инстанцией класса float (число с плавающей точкой короче)
    Значение best_value находится в диапазоне от 0 до 1.

    Мне в этом тесте не очень нравится то, что он проверяет некое очень узкое значение, в нашем случае итоговая метрика.
    Скорее вопрос к самой тестируемой функции. Много логики она делает. Хочется проверять некую общую логику.
    Не плохо, точно, просто вот такое ощущение. Хотя я никакой не тестировщик.
    """

    # 1. Генерация фиктивных данных для теста
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, n_redundant=2, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Создание объекта trial. Это имитация вызова данной функции в функции main(). Получается такая обертка над
    # основной логикой обучения, потому что дата-сет лежит в main.py, по этому и все циклы trial там пройдут
    def objective(trial):
        return random_forest_tuning(trial, X_train, y_train)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    # 3. Проверка результатов
    assert study.best_value is not None, "Функция должна возвращать значение метрики"
    assert isinstance(study.best_value, float), "Возвращаемое значение должно быть числом"
    assert 0 <= study.best_value <= 1, "Метрика accuracy должна быть в диапазоне [0, 1]"


def test_evaluate_model():
    '''Тестируемая функция evaluate_model() проекта wine_q.
    1. Должна сделать предсказания на обученной модели. Сгенерировать фиктивные данные, разделить их на тренировочные и
    оценочные
    2. Обучить модель на тренировочных данных.
    3. Вызовем тестируемую функцию
    4. Проверка возвращаемых значений с помощью pytest.approx — это инструмент в библиотеке pytest,
    который используется для проверки того, что два числа (или два набора чисел) близки друг к другу в пределах заданной точности
    (т.е. они не обязаны быть абсолютно равными). Мы сравнили метрики полученные внутри тестируемой функции и
    рассчитанные прямо здесь - те же самые метрики.
    5. Проверка диапазона значений метрик: F1_score,precision,recall.
    '''

    # 1. Генерация фиктивных данных для теста
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, n_redundant=2, random_state=42)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Обучение модели
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)

    # 3. Вызываем тестируемую функцию
    average_parametr = 'weighted'
    accuracy, f1, precision, recall = evaluate_model(model, X_valid, y_valid, aver=average_parametr)

    # Получаем предсказания модели здесь вручную, потому что в тестируемой функции они есть, но она их не возвращает
    y_pred = model.predict(X_valid)

    # 4. Проверка возвращаемых значений
    assert accuracy == pytest.approx(accuracy_score(y_valid, y_pred)), "Accuracy рассчитана неверно"
    assert f1 == pytest.approx(f1_score(y_valid, y_pred, average=average_parametr)), "F1-score рассчитана неверно"
    assert precision == pytest.approx(
        precision_score(y_valid, y_pred, average=average_parametr, zero_division=0.0)), "Precision рассчитана неверно"
    assert recall == pytest.approx(recall_score(y_valid, y_pred, average=average_parametr)), "Recall рассчитана неверно"

    # 5. Проверка диапазона значений метрик
    assert 0 <= accuracy <= 1, "Accuracy должна быть в диапазоне [0, 1]"
    assert 0 <= f1 <= 1, "F1-score должна быть в диапазоне [0, 1]"
    assert 0 <= precision <= 1, "Precision должна быть в диапазоне [0, 1]"
    assert 0 <= recall <= 1, "Recall должна быть в диапазоне [0, 1]"

