import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, LeakyReLU, Add, Conv1D, Flatten, \
    LSTM, Reshape, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import logging
import os
import optuna

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Функция загрузки данных
def load_data(year):
    logging.info(f"Загрузка данных за {year} год")
    data = pd.read_csv(f"{year}.csv")
    return data


# Функция обработки данных по городам
def process_data_clean_rf(data, year):
    logging.info(f"Обработка данных за {year} год")
    data_cleaned = data.iloc[:, [0, 6, 7]].rename(columns={
        data.columns[0]: "City",
        data.columns[6]: f"Students_{year}_Fulltime",
        data.columns[7]: f"Students_{year}_Parttime"
    })
    data_cleaned[f"Students_{year}_Fulltime"] = pd.to_numeric(data_cleaned[f"Students_{year}_Fulltime"],
                                                              errors="coerce")
    data_cleaned[f"Students_{year}_Parttime"] = pd.to_numeric(data_cleaned[f"Students_{year}_Parttime"],
                                                              errors="coerce")
    data_cleaned = data_cleaned.dropna().reset_index(drop=True)
    return data_cleaned


# Загрузка данных
years = [2020, 2021, 2022, 2023, 2024]
datasets = {year: load_data(year) for year in years}

# Обработка данных
processed_data = {year: process_data_clean_rf(datasets[year], year) for year in years}
final_data = processed_data[2020]
for year in years[1:]:
    final_data = pd.merge(final_data, processed_data[year], on="City", how="inner")

# Подготовка данных для нейросети
X = final_data[[f"Students_{year}_Fulltime" for year in years[:-1]]]
y = final_data[f"Students_{years[-1]}_Fulltime"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)


# Функция построения сложной гибридной модели (CNN + LSTM + Dense)
def build_complex_model(trial):
    filters = trial.suggest_int("filters", 32, 128, step=32)
    lstm_units = trial.suggest_int("lstm_units", 64, 256, step=64)
    dense_units = trial.suggest_int("dense_units", 128, 512, step=128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5, step=0.1)

    inputs = Input(shape=(X_train_scaled.shape[1], 1))
    x = Conv1D(filters=filters, kernel_size=2, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = LSTM(lstm_units, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Flatten()(x)
    x = Dense(dense_units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='mse', metrics=['mae'])
    return model


# Оптимизация гиперпараметров с Optuna
def objective(trial):
    model = build_complex_model(trial)
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test), verbose=0)
    loss, _ = model.evaluate(X_test_scaled, y_test, verbose=0)
    return loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_trial.params
logging.info(f"Лучшие гиперпараметры: {best_params}")

# Финальная модель с лучшими гиперпараметрами
model = build_complex_model(optuna.trial.FixedTrial(best_params))

# Коллбеки
log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
    TensorBoard(log_dir=log_dir, histogram_freq=1)
]

# Обучение модели
model.fit(X_train_scaled, y_train, epochs=1500, batch_size=64, validation_data=(X_test_scaled, y_test),
          callbacks=callbacks)

# Оценка модели
y_pred = model.predict(X_test_scaled).flatten()
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
logging.info(f"Оценка модели: MAE={mae}, MSE={mse}, R²={r2}")

# Визуализация результатов
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Фактическое количество студентов")
plt.ylabel("Предсказанное количество студентов")
plt.title("Сравнение предсказанных и фактических значений")
plt.show()

# Запуск TensorBoard
os.system(f"tensorboard --logdir={log_dir}")
