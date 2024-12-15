import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path)

    required_columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Отсутствуют необходимые колонки: {', '.join(missing_columns)}")

    X = data[['width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']].values
    y = data['class'].values
    y = pd.factorize(y)[0]  # Преобразуем классы в числовые значения

    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_dense_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # Предполагаем, что 10 классов
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))  # Предполагаем, что 10 классов
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping])
    return history.history

def create_models_and_train(data_path):
    X_train, X_test, y_train, y_test = load_data(data_path)
    input_shape = (X_train.shape[1],)

    # Model 1: Dense
    dense_model = create_dense_model(input_shape)
    dense_history = train_model(dense_model, X_train, y_train)

    # Model 2: CNN
    if X_train.shape[1] == 6:  # Ожидаем, что у нас 6 признаков
        X_train_cnn = np.random.rand(X_train.shape[0], 4, 3, 1)  # Псевдоданные, замените на настоящие
    else:
        raise ValueError("Неверная форма входных данных для CNN")

    cnn_model = create_cnn_model((4, 3, 1))
    cnn_history = train_model(cnn_model, X_train_cnn, y_train)

    return dense_history, cnn_history

# Пример использования
if __name__ == "__main__":
    data_path = 'RailData.csv'
    try:
        dense_history, cnn_history = create_models_and_train(data_path)
        print("Обучение моделей завершено успешно!")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
