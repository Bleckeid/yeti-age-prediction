# Определение возрастной категории пользователей РСЙ («Йети»)

Проект по разработке модели машинного обучения для автоматического определения возрастной категории анонимных пользователей на основе их поведения в сети. Это позволяет повысить точность таргетинга контекстной рекламы.

## Цели и задачи
- Создание конвейера (pipeline) для обработки сырых данных из различных источников.
- Генерация и отбор наиболее значимых признаков (Feature Engineering).
- Обучение модели многоклассовой классификации с достижением метрики **F1-macro >= 0.75**.
- Сохранение артефактов для дальнейшего внедрения (инференса).

## Стек 
- **Python 3.9+**
- **Pandas** (обработка данных и агрегация)
- **Scikit-learn** (Pipeline, ColumnTransformer, Feature Selection, Модели)
- **Joblib** (сериализация моделей)

## Структура 
- `model/` — папка с сохраненными артефактами (`model.pkl`, `preprocessor.pkl` и др.).
- `requirements.txt` — список зависимостей для запуска проекта.
- `.gitignore` — файл для исключения мусора и данных из репозитория.

## Как запустить 
Для получения предсказаний на новых данных используйте загруженные артефакты:

```python
import joblib

# Загрузка подготовленной функции и модели
prepare_features = joblib.load('model/prepare_features.pkl')
model = joblib.load('model/model.pkl')
preprocessor = joblib.load('model/preprocessor.pkl')
selected_features = joblib.load('model/selected_features.pkl')

# Обработка сырых данных
X_prepared = prepare_features(df_visits, df_activity, df_depth, df_device, df_usage)

# Предобработка и предсказание
X_transformed = preprocessor.transform(X_prepared)
# Фильтрация признаков перед предсказанием
X_final = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())[selected_features]

predictions = model.predict(X_final)