import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tkinter as tk
from tkinter import ttk


# Функция обработки данных с удалением лишних значений
def process_data_clean_rf(data, year):
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
    data_cleaned = data_cleaned[data_cleaned["City"].str.match(r"^[А-Яа-я\s\-]+$", na=False)]
    return data_cleaned


# Чтение данных из файлов
data_2022 = pd.read_csv('2022.csv')
data_2023 = pd.read_csv('2023.csv')
data_2024 = pd.read_csv('2024.csv')

# Обработка данных
data_2022_cleaned = process_data_clean_rf(data_2022, 2022)
data_2023_cleaned = process_data_clean_rf(data_2023, 2023)
data_2024_cleaned = process_data_clean_rf(data_2024, 2024)

# Объединение данных по городам
final_data = pd.merge(data_2022_cleaned, data_2023_cleaned, on="City", how="inner")
final_data = pd.merge(final_data, data_2024_cleaned, on="City", how="inner")

# Прогнозирование на 2025 год
X = final_data[['Students_2022_Fulltime', 'Students_2023_Fulltime', 'Students_2024_Fulltime']].values
y = final_data['Students_2024_Fulltime'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Компиляция модели
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Обучение модели
model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.2, verbose=1)

# Прогнозирование значений на 2025 год
final_data['Predicted_Students_2025_Fulltime'] = model.predict(scaler_X.transform(X)).flatten()
final_data['Predicted_Students_2025_Fulltime'] = scaler_y.inverse_transform(
    final_data['Predicted_Students_2025_Fulltime'].values.reshape(-1, 1))

# Загрузка данных по образовательным программам
pred_files = ["pred2021.csv", "pred2022.csv", "pred2023.csv", "pred2024.csv"]
pred_data_list = []

columns_needed = ["Программа", "Код", "за счет бюджет-ных ассигнова-ний",
                  "по договорам об оказании платных образова-тельных услуг"]

for file in pred_files:
    try:
        df = pd.read_csv(file)
        existing_columns = [col for col in columns_needed if col in df.columns]

        if existing_columns:
            df = df[columns_needed]
            df["Год"] = file.replace("pred", "").replace(".csv", "")
            pred_data_list.append(df)
    except Exception as e:
        print(f"Ошибка при обработке файла {file}: {e}")

# Объединяем все данные по программам образования
if pred_data_list:
    pred_data = pd.concat(pred_data_list, ignore_index=True)
    for col in columns_needed[2:]:
        if col in pred_data.columns:
            pred_data[col] = pd.to_numeric(pred_data[col], errors="coerce").fillna(0)

    # Прогноз на 2025 год для программ
    X_prog = pred_data[
        ["за счет бюджет-ных ассигнова-ний", "по договорам об оказании платных образова-тельных услуг"]].values
    scaler_prog = StandardScaler()
    X_prog_scaled = scaler_prog.fit_transform(X_prog)

    pred_data["Predicted_2025_Budget"] = model.predict(X_prog_scaled)[:, 0] * scaler_prog.scale_[0] + scaler_prog.mean_[
        0]
    pred_data["Predicted_2025_Paid"] = model.predict(X_prog_scaled)[:, 0] * scaler_prog.scale_[1] + scaler_prog.mean_[1]

    pred_data[["Predicted_2025_Budget", "Predicted_2025_Paid"]] = pred_data[
        ["Predicted_2025_Budget", "Predicted_2025_Paid"]].round(0).astype(int)

# Сохранение итогового датасета в xlsx
output_xlsx = 'predicted_data_2025.xlsx'
with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
    final_data.to_excel(writer, sheet_name="Regions", index=False)
    if pred_data_list:
        pred_data.to_excel(writer, sheet_name="Programs", index=False)


# Функция для отображения адаптивного окна с таблицей
def show_table(df, title="Данные"):
    root = tk.Tk()
    root.title(title)

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)

    tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")

    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=150, anchor="center")

    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    tree.pack(fill="both", expand=True)

    root.geometry("1200x600")
    root.mainloop()


# Вызов функции отображения таблицы
show_table(final_data, "Прогноз по регионам и образовательным программам на 2025 год")


# Функция для прогноза количества обучающихся на заочной форме обучения
def predict_parttime_students(model, data):
    predicted_values = model.predict(data)
    return predicted_values


# Функция для отображения прогноза в всплывающем окне
def show_parttime_prediction(model, data):
    prediction = predict_parttime_students(model, data)
    root = tk.Tk()
    root.title("Прогноз заочной формы обучения")

    label = tk.Label(root, text=f"Прогнозируемое количество обучающихся на заочной форме: {prediction[0]:.2f}")
    label.pack(padx=10, pady=10)

    button = tk.Button(root, text="Закрыть", command=root.destroy)
    button.pack(pady=5)

    root.mainloop()


# Функция для анализа поступающих по программам обучения
def analyze_programs(file_path):
    data = pd.read_csv(file_path, skiprows=12)  # Пропускаем 12 строк, начинаем с 13-й
    programs_analysis = data.iloc[:, [0, 2, 3, 7]].rename(columns={
        data.columns[0]: "Program Name",
        data.columns[2]: "Program Code",
        data.columns[3]: "Budget Students",
        data.columns[7]: "Paid Students"
    })
    return programs_analysis


# Функция для отображения анализа поступающих в всплывающем окне
def show_program_analysis():
    data_2023 = analyze_programs("pred2023.csv")
    data_2024 = analyze_programs("pred2024.csv")

    root = tk.Tk()
    root.title("Анализ поступающих по программам")

    tree = ttk.Treeview(root, columns=("Program Name", "Program Code", "Budget Students", "Paid Students"),
                        show="headings")
    tree.heading("Program Name", text="Название программы")
    tree.heading("Program Code", text="Код направления")
    tree.heading("Budget Students", text="Бюджетная форма")
    tree.heading("Paid Students", text="Платная форма")

    for _, row in data_2023.iterrows():
        tree.insert("", "end", values=row.tolist())

    for _, row in data_2024.iterrows():
        tree.insert("", "end", values=row.tolist())

    tree.pack(expand=True, fill="both")

    button = tk.Button(root, text="Закрыть", command=root.destroy)
    button.pack(pady=5)

    root.mainloop()


# Исправленная функция анализа программ обучения
def analyze_programs_fixed(file_path):
    try:
        data = pd.read_csv(file_path, skiprows=12)
        if data.shape[1] < 8:
            raise ValueError("Недостаточно колонок в файле.")

        programs_analysis = data.iloc[:, [0, 2, 3, 7]].rename(columns={
            data.columns[0]: "Program Name",
            data.columns[2]: "Program Code",
            data.columns[3]: "Budget Students",
            data.columns[7]: "Paid Students"
        })
        return programs_analysis
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")
        return None


# Функция для отображения анализа поступающих в исправленном варианте
def show_program_analysis_fixed():
    global data_2023_fixed, data_2024_fixed

    if data_2023_fixed is None or data_2024_fixed is None:
        print("Ошибка: данные не загружены.")
        return

    root = tk.Tk()
    root.title("Анализ поступающих по программам")

    columns = ("Program Name", "Program Code", "Budget Students", "Paid Students")
    tree = ttk.Treeview(root, columns=columns, show="headings")

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=150)

    for _, row in data_2023_fixed.iterrows():
        tree.insert("", "end", values=row.tolist())

    for _, row in data_2024_fixed.iterrows():
        tree.insert("", "end", values=row.tolist())

    tree.pack(expand=True, fill="both")

    button = tk.Button(root, text="Закрыть", command=root.destroy)
    button.pack(pady=5)

    root.mainloop()

import pandas as pd

# Пути к файлам
file_2023 = "pred2023.csv"
file_2024 = "pred2024.csv"
output_file = "predictions_2025.xlsx"

# Читаем данные из CSV, начиная с 13-й строки (12-й индекс в pandas)
df_2023 = pd.read_csv(file_2023, skiprows=12)
df_2024 = pd.read_csv(file_2024, skiprows=12)

# Выбираем нужные столбцы (название специальности, код, бюджет, платная форма)
columns_needed = [0, 2, 3, 7]  # Индексы нужных столбцов
df_2023 = df_2023.iloc[:, columns_needed]
df_2024 = df_2024.iloc[:, columns_needed]

# Переименовываем столбцы для удобства
column_names = ["Специальность", "Код", "Бюджет", "Платно"]
df_2023.columns = column_names
df_2024.columns = column_names

# Простая модель прогноза: усреднение показателей за 2023 и 2024 годы
predictions = df_2023.copy()
predictions["Бюджет"] = (df_2023["Бюджет"] + df_2024["Бюджет"]) // 2
predictions["Платно"] = (df_2023["Платно"] + df_2024["Платно"]) // 2

# Добавляем прогноз как 2025 год
predictions.insert(0, "Год", 2025)

# Формируем окончательный датафрейм
final_df = pd.DataFrame()
final_df["Специальность"] = df_2023["Специальность"]
final_df["Код"] = df_2023["Код"]
final_df["Бюджет 2023"] = df_2023["Бюджет"]
final_df["Платно 2023"] = df_2023["Платно"]
final_df["Бюджет 2024"] = df_2024["Бюджет"]
final_df["Платно 2024"] = df_2024["Платно"]
final_df["Прогноз Бюджет 2025"] = predictions["Бюджет"]
final_df["Прогноз Платно 2025"] = predictions["Платно"]


# Интеграция с основным кодом
def integrate_predictions(existing_df):
    return existing_df.merge(final_df, on=["Специальность", "Код"], how="left")


# Записываем данные в Excel
with pd.ExcelWriter(output_file) as writer:
    final_df.to_excel(writer, sheet_name="Прогноз 2025", index=False)

print(f"Файл с прогнозами сохранён: {output_file}")

import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog

# Пути к файлам
file_2023 = "pred2023.csv"
file_2024 = "pred2024.csv"
output_file = "predictions_2025.xlsx"

# Читаем данные из CSV, начиная с 13-й строки (12-й индекс в pandas)
df_2023 = pd.read_csv(file_2023, skiprows=12)
df_2024 = pd.read_csv(file_2024, skiprows=12)

# Выбираем нужные столбцы (название специальности, код, бюджет, платная форма)
columns_needed = [0, 2, 3, 7]  # Индексы нужных столбцов
df_2023 = df_2023.iloc[:, columns_needed]
df_2024 = df_2024.iloc[:, columns_needed]

# Переименовываем столбцы для удобства
column_names = ["Специальность", "Код", "Бюджет", "Платно"]
df_2023.columns = column_names
df_2024.columns = column_names

# Простая модель прогноза: усреднение показателей за 2023 и 2024 годы
predictions = df_2023.copy()
predictions["Бюджет"] = (df_2023["Бюджет"] + df_2024["Бюджет"]) // 2
predictions["Платно"] = (df_2023["Платно"] + df_2024["Платно"]) // 2

# Добавляем прогноз как 2025 год
predictions.insert(0, "Год", 2025)

# Формируем окончательный датафрейм
final_df = pd.DataFrame()
final_df["Специальность"] = df_2023["Специальность"]
final_df["Код"] = df_2023["Код"]
final_df["Бюджет 2023"] = df_2023["Бюджет"]
final_df["Платно 2023"] = df_2023["Платно"]
final_df["Бюджет 2024"] = df_2024["Бюджет"]
final_df["Платно 2024"] = df_2024["Платно"]
final_df["Прогноз Бюджет 2025"] = predictions["Бюджет"]
final_df["Прогноз Платно 2025"] = predictions["Платно"]


# Графический интерфейс
class DataApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Просмотр данных")

        self.notebook = ttk.Notebook(root)
        self.page1 = ttk.Frame(self.notebook)
        self.page2 = ttk.Frame(self.notebook)

        self.notebook.add(self.page1, text="Основные данные")
        self.notebook.add(self.page2, text="Добавленные данные")

        self.notebook.pack(expand=True, fill="both")

        # Основные данные (первая страница)
        self.tree1 = ttk.Treeview(self.page1)
        self.tree1.pack(expand=True, fill="both")

        # Вторая страница - кнопка для добавления данных
        self.add_data_button = tk.Button(self.page2, text="Добавить данные из файла", command=self.load_file)
        self.add_data_button.pack()

        self.tree2 = ttk.Treeview(self.page2)
        self.tree2.pack(expand=True, fill="both")

        # Кнопка для выхода
        self.quit_button = tk.Button(root, text="Выход", command=root.quit)
        self.quit_button.pack()

        self.load_main_data()

    def load_main_data(self):
        try:
            self.populate_treeview(self.tree1, final_df)
        except Exception as e:
            print("Ошибка загрузки основных данных:", e)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx;*.xls")])
        if file_path:
            try:
                df = pd.read_excel(file_path)
                self.populate_treeview(self.tree2, df)
            except Exception as e:
                print("Ошибка загрузки файла:", e)

    def populate_treeview(self, tree, df):
        tree.delete(*tree.get_children())

        tree["columns"] = list(df.columns)
        tree["show"] = "headings"

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row))


if __name__ == "__main__":
    root = tk.Tk()
    app = DataApp(root)
    root.mainloop()

import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog

# Пути к файлам
file_2023 = "pred2023.csv"
file_2024 = "pred2024.csv"
output_file = "predictions_2025.xlsx"

# Читаем данные из CSV, начиная с 13-й строки (12-й индекс в pandas)
df_2023 = pd.read_csv(file_2023, skiprows=12)
df_2024 = pd.read_csv(file_2024, skiprows=12)

# Выбираем нужные столбцы (название специальности, код, бюджет, платная форма)
columns_needed = [0, 2, 3, 7]  # Индексы нужных столбцов
df_2023 = df_2023.iloc[:, columns_needed]
df_2024 = df_2024.iloc[:, columns_needed]

# Переименовываем столбцы для удобства
column_names = ["Специальность", "Код", "Бюджет", "Платно"]
df_2023.columns = column_names
df_2024.columns = column_names

# Простая модель прогноза: усреднение показателей за 2023 и 2024 годы
predictions = df_2023.copy()
predictions["Бюджет"] = ((df_2023["Бюджет"].fillna(0) + df_2024["Бюджет"].fillna(0)) // 2).fillna(0).astype(int)
predictions["Платно"] = ((df_2023["Платно"].replace([np.inf, -np.inf], np.nan).fillna(0) + df_2024["Платно"].replace([np.inf, -np.inf], np.nan).fillna(0)) // 2).fillna(0).astype(int)

# Добавляем прогноз как 2025 год
predictions.insert(0, "Год", 2025)

# Формируем окончательный датафрейм
final_df = pd.DataFrame()
final_df["Специальность"] = df_2023["Специальность"]
final_df["Код"] = df_2023["Код"]
final_df["Бюджет 2023"] = df_2023["Бюджет"].replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
final_df["Платно 2023"] = df_2023["Платно"].astype(int)
final_df["Бюджет 2024"] = df_2024["Бюджет"].astype(int)
final_df["Платно 2024"] = df_2024["Платно"].astype(int)
final_df["Прогноз Бюджет 2025"] = predictions["Бюджет"]
final_df["Прогноз Платно 2025"] = predictions["Платно"]


# Графический интерфейс
class DataApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Просмотр данных")

        self.notebook = ttk.Notebook(root)
        self.page1 = ttk.Frame(self.notebook)
        self.page2 = ttk.Frame(self.notebook)

        self.notebook.add(self.page1, text="Города")
        self.notebook.add(self.page2, text="Специальности")

        self.notebook.pack(expand=True, fill="both")

        # Города (первая страница)
        self.tree1 = ttk.Treeview(self.page1)
        self.tree1.pack(expand=True, fill="both")

        # Специальности (вторая страница)
        self.tree2 = ttk.Treeview(self.page2)
        self.tree2.pack(expand=True, fill="both")

        # Кнопка для выхода
        self.quit_button = tk.Button(root, text="Выход", command=root.quit)
        self.quit_button.pack()

        self.load_data()

    def load_data(self):
        try:
            # Разделяем данные по первой и второй страницам
            city_data = final_df[["Специальность", "Код"]]  # Данные для городов
            specialty_data = final_df  # Все данные о специальностях

            self.populate_treeview(self.tree1, city_data)
            self.populate_treeview(self.tree2, specialty_data)
        except Exception as e:
            print("Ошибка загрузки данных:", e)

    def populate_treeview(self, tree, df):
        tree.delete(*tree.get_children())

        tree["columns"] = list(df.columns)
        tree["show"] = "headings"

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row.astype(str)))


if __name__ == "__main__":
    root = tk.Tk()
    app = DataApp(root)
    root.mainloop()
