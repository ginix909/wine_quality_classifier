import os
import sys
import pandas as pd

from ML.Aicrowd.wine_q.src.data_processing import split_data

# Добавляем корневую папку проекта в пути поиска
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


def test_split_data():
    # Пример данных
    data = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8], "target": [0, 1, 0, 1]})
    X_train, X_valid, y_train, y_valid = split_data(data, target_column="target")
    assert len(X_train) > 0 and len(X_valid) > 0, "Разделение данных должно быть корректным"
    assert "target" not in X_train.columns, "Целевая переменная не должна быть в признаках"


