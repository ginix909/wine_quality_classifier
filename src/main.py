from pathlib import Path
import optuna

from data_processing import load_data, preprocess_data, split_data, encode_features
from model import evaluate_model, make_predictions, random_forest_tuning
from sklearn.ensemble import RandomForestClassifier


def main():
    # Шаг 1: Загрузка данных
    print("Шаг 1: Загрузка тренировочных и тестовых данных...")
    train = load_data(file_name='train.csv')
    test = load_data(file_name='test.csv')

    # Шаг 2: Предварительная Обработка данных: train,test. Удаление пропусков, построение новых признаков, удаление ненужных.
    print("Шаг 2: Предварительная обработка данных (первый этап грубой обработки)...")
    processed_train = preprocess_data(train)
    processed_test = preprocess_data(test)

    # Шаг 3: Разделение тренировочного (train) набора на train и valid
    print("Шаг 3: Разделение тренировочного набора на train,valid.")
    X_train, X_valid, y_train, y_valid = split_data(processed_train, target_column="quality")

    # Шаг 4: Кодирование признаков (только на основе тренировочного набора)
    print("Шаг 4: Кодирование категориальных признаков в наборах: train,valid,test.")
    X_train, X_valid, X_test = encode_features(X_train, X_valid, processed_test)


    # Шаг 5: поиск лучших параметров. Возвращает среднее значение параметров перекрестной проверки
    # (cross-validated score) из модели RandomForestClassifier после всех итераций (trials).
    print("Шаг 5: Оптимизация гиперпараметров...")
    # Объект study - задает направления обучения (максимизировать/минимизировать), хранит данные о всех запусках (trials)
    # так как парметр scoring=accuracy, то есть мы считаем точность во время кросс-валидации, а точность нужна чем выше,
    # тем лучше, значит направление обучения это максимизация.
    # study = optuna.create_study(direction='maximize')
    # Надстроил над своей функцией еще безымянную функцию, потому что optimize() должна принимать только trial, она только
    # trial и примет, передаст в random_forest_tuning(), помимо двух уже имеющихся там аргументов.
    # study.optimize(lambda trial: random_forest_tuning(trial, X_train, y_train), n_trials=50)

    # Лучшие гиперпараметры
    # best_params = study.best_params
    # print("Лучшие гиперпараметры:", best_params)
    best_params = dict({'criterion': 'entropy', 'n_estimators': 94, 'max_depth': 19, 'min_samples_split': 7, 'min_samples_leaf': 4})

    # Шаг 6: Обучение модели
    print("Шаг 6: Обучение модели на лучших параметрах...")
    # Создание модели с лучшими гиперпараметрами. Передаем распакованный словарь **best_params
    model = RandomForestClassifier(**best_params, random_state=1)
    model.fit(X_train, y_train)


    # Шаг 7: Оценка модели на валидационном наборе
    print("Шаг 7: Оценка модели...")
    # valid_results = evaluate_model(model, X_valid, y_valid)
    # print("Результаты на валидационном наборе:", valid_results)
    average_parametr = 'weighted'
    accuracy, f1, precision, recall = evaluate_model(model, X_valid,y_valid, aver=average_parametr)

    print(f"Accuracy модели с лучшими фиксированными параметрами: {round(accuracy,3)}")
    print(f"f1_macro модели с лучшими фиксированными параметрами: {round(f1,3)}")
    print(f"Classification Report. Отчет по модели {model.__class__.__name__}.")
    # print(report)
    print(f'Параметр вычисления среднего для метрик precision, recall: {average_parametr}')
    print(f'Точность precision: {round(precision,3)}')
    print(f'Полнота recall: {round(recall,3)}')
    # average=weighted:
    # Точность precision: 0.621
    # Полнота recall: 0.627

    # # Шаг 8: Итоговые предсказания на тестовом наборе
    print("Шаг 8: Предсказания на тестовом наборе, получаем дата-фрейм...")
    predictions = make_predictions(model, X_test)
    print("Предсказания для тестового набора готовы.")

    # Шаг 9: Сохранение предсказаний
    print("Шаг 9: Сохранение предсказаний...")
    # Относительный Путь к папке data
    data_dir = Path("../data")
    predictions_file = data_dir / "wine_submissions.csv" # предсказания винных категориий (предстваления)
    predictions.to_csv(predictions_file, index=False)
    print(f"Предсказания сохранены в файл '{predictions_file}'.")


if __name__ == "__main__":
    main()

