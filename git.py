import pandas as pd

def process_data_clean_rf(data, year):
    data_cleaned = data.iloc[:, [0, 6]].rename(columns={
        data.columns[0]: "City",
        data.columns[6]: f"Students_{year}"
    })
    data_cleaned[f"Students_{year}"] = pd.to_numeric(data_cleaned[f"Students_{year}"], errors="coerce")
    data_cleaned = data_cleaned.dropna().reset_index(drop=True)

    data_cleaned = data_cleaned[data_cleaned["City"].str.match(r"^[А-Яа-я\s\-]+$", na=False)]

    rf_row = data_cleaned[data_cleaned["City"].str.contains("Российская Федерация", na=False)]
    data_cleaned = data_cleaned[~data_cleaned["City"].str.contains("Российская Федерация", na=False)]
    return data_cleaned, rf_row

data_2022 = pd.read_csv('2022.csv')
data_2023 = pd.read_csv('2023.csv')
data_2024 = pd.read_csv('2024.csv')

data_2022_cleaned, rf_2022 = process_data_clean_rf(data_2022, 2022)
data_2023_cleaned, rf_2023 = process_data_clean_rf(data_2023, 2023)
data_2024_cleaned, rf_2024 = process_data_clean_rf(data_2024, 2024)

merged_data = pd.merge(data_2022_cleaned, data_2023_cleaned, on="City", how="inner")
merged_data = pd.merge(merged_data, data_2024_cleaned, on="City", how="inner")

rf_data = pd.merge(rf_2022, rf_2023, on="City", how="inner")
rf_data = pd.merge(rf_data, rf_2024, on="City", how="inner")
rf_data = rf_data.iloc[:1]
final_data = pd.concat([rf_data, merged_data], ignore_index=True)

final_data.to_csv('merged_data_with_rf_cleaned.csv', index=False)
print("Файл сохранён как 'merged_data_with_rf_cleaned.csv'. Лишние строки удалены.")

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('merged_data_with_rf_cleaned.csv')
X = data[['Students_2022', 'Students_2023', 'Students_2024']].values
y = data['Students_2024'].values

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, data.index, test_size=0.2, random_state=42, shuffle=True
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.2, verbose=1)
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Средняя абсолютная ошибка (MAE) на тестовой выборке: {test_mae}")

predictions_scaled = model.predict(X_test).flatten()
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
mae = mean_absolute_error(y_test_original, predictions)
rmse = np.sqrt(mean_squared_error(y_test_original, predictions))

percent_errors = np.abs((y_test_original - predictions) / y_test_original) * 100
mean_percent_error = percent_errors.mean()
accuracy = 100 - mean_percent_error

print(f"Средняя абсолютная ошибка (MAE): {mae}")
print(f"Среднеквадратичная ошибка (RMSE): {rmse}")
print(f"Средняя точность прогноза: {accuracy:.2f}%")

data['Predicted_Students_2025'] = model.predict(scaler_X.transform(X)).flatten()
data['Predicted_Students_2025'] = scaler_y.inverse_transform(data['Predicted_Students_2025'].values.reshape(-1, 1))

output_file = 'predicted_students_simplified_model.xlsx'
data[['City', 'Students_2022', 'Students_2023', 'Students_2024', 'Predicted_Students_2025']].to_excel(output_file, index=False)

print(f"Документ с улучшенными прогнозами создан: {output_file}")
