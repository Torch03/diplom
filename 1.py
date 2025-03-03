import wx
import wx.grid as gridlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('WXAgg')
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from sklearn.metrics import r2_score
import time
import threading
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, Image
from reportlab.lib import colors
import os
import pickle

import platform
if platform.system() == 'Darwin':
    wx.SystemOptions.SetOption("osx.openfiledialog.always-show-types", "1")
    wx.SystemOptions.SetOption("osx.menubar.allow-in-nsapp", "1")


# =================================
# Prediction Tab Implementation
# =================================
class PredictionTab(wx.Panel):
    title = "Прогнозы"

    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.predictions = None
        self.init_ui()
        self.SetBackgroundColour(wx.Colour(255, 255, 255))  # macOS background

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Таблица прогнозов
        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 6)
        self.grid.SetSelectionMode(gridlib.Grid.GridSelectRows)

        # Настройка колонок
        columns = [
            ("Специальность", 300, wx.ALIGN_LEFT),
            ("Текущий бюджет", 120, wx.ALIGN_RIGHT),
            ("Прогноз", 120, wx.ALIGN_RIGHT),
            ("Изменение (%)", 100, wx.ALIGN_CENTER),
            ("Рекомендация", 200, wx.ALIGN_LEFT),
            ("Уровень риска", 100, wx.ALIGN_CENTER)
        ]

        for col, (label, width, align) in enumerate(columns):
            self.grid.SetColLabelValue(col, label)
            self.grid.SetColSize(col, width)
            self.grid.SetColLabelAlignment(align, wx.ALIGN_CENTER)
            attr = gridlib.GridCellAttr()
            attr.SetAlignment(align, wx.ALIGN_CENTER)
            self.grid.SetColAttr(col, attr)

        # Панель управления
        control_panel = wx.Panel(self)
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        # Кнопки экспорта
        btn_excel = wx.Button(control_panel, label="Excel", size=(120, 30))
        btn_excel.Bind(wx.EVT_BUTTON, self.on_export_excel)

        btn_pdf = wx.Button(control_panel, label="PDF", size=(120, 30))
        btn_pdf.Bind(wx.EVT_BUTTON, self.on_export_pdf)

        hbox.Add(btn_excel, 0, wx.RIGHT, 10)
        hbox.Add(btn_pdf, 0)
        control_panel.SetSizer(hbox)

        vbox.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        vbox.Add(control_panel, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        self.SetSizer(vbox)

        # macOS оптимизация
        if wx.Platform == '__WXMAC__':
            self.grid.SetLabelFont(wx.Font(12, wx.FONTFAMILY_DEFAULT,
                                           wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            self.grid.SetDefaultCellFont(wx.Font(12, wx.FONTFAMILY_DEFAULT,
                                                 wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

    def update_predictions(self, data, predictions):
        """Обновление таблицы прогнозов"""
        self.grid.ClearGrid()
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())

        for i, row in data.iterrows():
            self.grid.AppendRows(1)
            pred_value = predictions[i][0]
            change = ((pred_value - row['budget']) / row['budget']) * 100

            # Цветовое кодирование
            if change > 10:
                color = wx.Colour(220, 255, 220)
            elif change < -5:
                color = wx.Colour(255, 220, 220)
            else:
                color = wx.Colour(255, 255, 255)

            self.grid.SetCellValue(i, 0, row['specialty'])
            self.grid.SetCellValue(i, 1, f"{row['budget']:,.1f}")
            self.grid.SetCellValue(i, 2, f"{pred_value:,.1f}")
            self.grid.SetCellValue(i, 3, f"{change:.1f}%")
            self.grid.SetCellValue(i, 4, "Увеличить финансирование" if change > 10 else
            "Сократить набор" if change < -5 else "Стабильный")
            self.grid.SetCellValue(i, 5, "Высокий" if abs(change) > 20 else
            "Средний" if abs(change) > 10 else "Низкий")

            # Установка цвета фона
            for col in range(6):
                self.grid.SetCellBackgroundColour(i, col, color)

    def on_export_excel(self, event):
        """Экспорт в Excel"""
        pass  # Реализация экспорта

    def on_export_pdf(self, event):
        """Экспорт в PDF"""
        pass  # Реализация экспорта

    # =================================
    # Graph Tab Implementation
    # =================================
    class GraphTab(wx.Panel):
        title = "Аналитика"

        def __init__(self, parent, main_frame):
            super().__init__(parent)
            self.main_frame = main_frame
            self.init_ui()
            self.SetBackgroundColour(wx.Colour(255, 255, 255))  # macOS background

        def init_ui(self):
            vbox = wx.BoxSizer(wx.VERTICAL)

            # Набор графиков
            self.figure = Figure(figsize=(10, 6), dpi=100)
            self.canvas = FigureCanvas(self, -1, self.figure)

            # Панель управления
            control_panel = wx.Panel(self)
            hbox = wx.BoxSizer(wx.HORIZONTAL)

            self.btn_prev = wx.Button(control_panel, label="← Назад", size=(80, 30))
            self.btn_next = wx.Button(control_panel, label="Вперед →", size=(80, 30))

            hbox.Add(self.btn_prev, 0, wx.RIGHT, 10)
            hbox.Add(self.btn_next, 0)
            control_panel.SetSizer(hbox)

            vbox.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
            vbox.Add(control_panel, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)
            self.SetSizer(vbox)

            # macOS оптимизация
            if wx.Platform == '__WXMAC__':
                self.btn_prev.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT,
                                              wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
                self.btn_next.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT,
                                              wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

        def update_graphs(self, data, predictions):
            """Обновление графиков"""
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Пример графика
            changes = ((predictions[:, 0] - data['budget'].values) / data['budget'].values) * 100
            ax.hist(changes, bins=20, color='skyblue', edgecolor='black')
            ax.set_title('Распределение изменений бюджета', fontsize=14)
            ax.set_xlabel('Процент изменения', fontsize=12)
            ax.set_ylabel('Количество специальностей', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)

            self.canvas.draw()

# =================================
# Neural Network Implementation
# =================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_size=128, output_size=4):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size // 2) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, hidden_size // 2))
        self.W3 = np.random.randn(hidden_size // 2, output_size) * np.sqrt(2. / (hidden_size // 2))
        self.b3 = np.zeros((1, output_size))
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3

    def train(self, X, y, epochs=2000, lr=0.0001, batch_size=64, progress_callback=None):
        # Проверка данных
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Ошибка в тренировочных данных: NaN или бесконечные значения")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("Ошибка в целевых данных: NaN или бесконечные значения")

        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-8
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0) + 1e-8

        X_norm = (X - self.X_mean) / self.X_std
        y_norm = (y - self.y_mean) / self.y_std

        split_idx = int(0.8 * X.shape[0])
        X_train, X_val = X_norm[:split_idx], X_norm[split_idx:]
        y_train, y_val = y_norm[:split_idx], y_norm[split_idx:]

        best_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self._train_batch(X_train, y_train, lr, batch_size)
            val_loss = self._validate(X_val, y_val)

            if progress_callback:
                progress = int((epoch + 1) / epochs * 100)
                wx.CallAfter(progress_callback, progress,
                           f"Epoch {epoch + 1}/{epochs}\nTrain Loss: {train_loss:.4f}\nVal Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
            elif val_loss > best_loss * 1.1:
                break

    def _train_batch(self, X, y, lr, batch_size):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        total_loss = 0

        for i in range(0, X.shape[0], batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            output = self.forward(X_batch)
            loss = np.mean((output - y_batch) ** 2)
            total_loss += loss * len(batch_indices)

            delta3 = (output - y_batch) / batch_size
            dW3 = np.dot(self.a2.T, delta3)
            db3 = np.sum(delta3, axis=0, keepdims=True)

            delta2 = np.dot(delta3, self.W3.T) * (self.z2 > 0)
            dW2 = np.dot(self.a1.T, delta2)
            db2 = np.sum(delta2, axis=0, keepdims=True)

            delta1 = np.dot(delta2, self.W2.T) * (self.z1 > 0)
            dW1 = np.dot(X_batch.T, delta1)
            db1 = np.sum(delta1, axis=0, keepdims=True)

            self.W3 -= lr * dW3
            self.b3 -= lr * db3
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

        return total_loss / X.shape[0]

    def _validate(self, X, y):
        output = self.forward(X)
        return np.mean((output - y) ** 2)

    def predict(self, X):
        X_norm = (X - self.X_mean) / self.X_std
        y_norm = self.forward(X_norm)
        return y_norm * self.y_std + self.y_mean

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.__dict__ = pickle.load(f)


# =================================
# Loading Screen Implementation
# =================================
class LoadingScreen(wx.Dialog):
    def __init__(self, parent):
        super().__init__(parent, title="Training Progress", size=(400, 200))
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.gauge = wx.Gauge(panel, range=100, size=(300, 25))
        self.status = wx.StaticText(panel, label="Initializing training...")

        vbox.Add(self.gauge, 0, wx.ALL | wx.EXPAND, 10)
        vbox.Add(self.status, 0, wx.ALL | wx.EXPAND, 10)
        panel.SetSizer(vbox)

        self.Centre()
        self.Show()

    def update(self, value, message):
        self.gauge.SetValue(value)
        self.status.SetLabel(message)
        wx.Yield()



# =================================
# Main Application Window
# =================================
class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Прогноз приема в вузы", size=(1200, 800))

        if platform.system() == 'Darwin':
            self.SetBackgroundColour(wx.Colour(255, 255, 255))
            self.SetWindowVariant(wx.WINDOW_VARIANT_NORMAL)

        self.data = {}
        self.nn = None
        self.current_predictions = None
        self.init_ui()

        if platform.system() == 'Darwin':
            self.SendSizeEvent()
            self.Update()
            self.Refresh()

    def init_ui(self):
        main_panel = wx.Panel(self)
        main_panel.SetBackgroundColour(wx.Colour(255, 255, 255))

        notebook = wx.Notebook(main_panel)
        notebook.SetBackgroundColour(wx.Colour(255, 255, 255))

        self.tabs = {
            'input': InputTab(notebook, self),
            'predict': PredictionTab(notebook, self),
            'future': FuturePredictionTab(notebook, self),
            'graphs': GraphTab(notebook, self)
        }

        for name, tab in self.tabs.items():
            notebook.AddPage(tab, tab.title)
            if platform.system() == 'Darwin':
                tab.SetBackgroundColour(wx.Colour(255, 255, 255))

        self.setup_toolbar()
        self.statusbar = self.CreateStatusBar(2)

        if platform.system() == 'Darwin':
            self.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT,
                                 wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            self.statusbar.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT,
                                           wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 5)
        main_panel.SetSizer(sizer)

        self.Centre()
        self.Show(True)
        self.Refresh()

    def setup_toolbar(self):
        self.toolbar = self.CreateToolBar()
        train_tool = self.toolbar.AddTool(wx.ID_ANY, "Обучить", wx.ArtProvider.GetBitmap(wx.ART_PLUS))
        predict_tool = self.toolbar.AddTool(wx.ID_ANY, "Прогноз", wx.ArtProvider.GetBitmap(wx.ART_FIND))
        self.toolbar.Realize()

        self.Bind(wx.EVT_TOOL, self.on_train, train_tool)
        self.Bind(wx.EVT_TOOL, self.on_predict, predict_tool)

    def on_train(self, event):
        if not self.data:
            wx.MessageBox("Нет данных для обучения", "Ошибка", wx.OK | wx.ICON_ERROR)
            return

        dlg = wx.ProgressDialog("Обучение модели", "Подготовка...", maximum=100, parent=self)

        def train_thread():
            try:
                X, y = self.prepare_training_data()
                self.nn = NeuralNetwork(X.shape[1])
                self.nn.train(X, y, progress_callback=lambda p, m: wx.CallAfter(dlg.Update, p, m))
                wx.CallAfter(dlg.Destroy)
                wx.CallAfter(self.update_status, "Обучение завершено")
            except Exception as e:
                wx.CallAfter(dlg.Destroy)
                wx.CallAfter(wx.MessageBox, f"Ошибка обучения: {str(e)}", "Ошибка", wx.OK | wx.ICON_ERROR)

        threading.Thread(target=train_thread).start()

    def on_predict(self, event):
        if not self.nn or not self.data:
            wx.MessageBox("Сначала обучите модель", "Ошибка", wx.OK | wx.ICON_ERROR)
            return

        try:
            latest_year = sorted(self.data.keys())[-1]
            df = self.data[latest_year]
            X = df[['budget', 'target', 'quota', 'paid']].values
            predictions = self.nn.predict(X)

            # Обработка прогнозов
            self.current_predictions = np.maximum(np.floor(predictions), 0)  # Округление и замена отрицательных

            self.tabs['predict'].update_predictions(df, self.current_predictions)
            self.tabs['graphs'].update_graphs()  # Исправлен вызов метода

            wx.MessageBox("Прогноз успешно сгенерирован", "Успех", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(f"Ошибка прогнозирования: {str(e)}", "Ошибка", wx.OK | wx.ICON_ERROR)

    def prepare_training_data(self):
        if len(self.data) < 2:
            raise ValueError("Нужны данные как минимум за 2 года")

        features = []
        targets = []
        years = sorted(self.data.keys())

        for i in range(len(years) - 1):
            current = self.data[years[i]]
            next_year = self.data[years[i + 1]]

            merged = pd.merge(
                current, next_year,
                on='specialty',
                suffixes=('_current', '_next')
            )

            features.append(merged[['budget_current', 'target_current',
                                    'quota_current', 'paid_current']].values)
            targets.append(merged[['budget_next', 'target_next',
                                   'quota_next', 'paid_next']].values)

        return np.vstack(features), np.vstack(targets)

    def update_status(self, message):
        self.statusbar.SetStatusText(message, 0)


# =================================
# Input Tab Implementation
# =================================
class InputTab(wx.Panel):
    title = "Входные данные"

    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetBackgroundColour(wx.Colour(255, 255, 255))

        control_panel = wx.Panel(self)
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        self.year_entry = wx.TextCtrl(control_panel, size=(100, -1))
        btn_paste = wx.Button(control_panel, label="Вставить из Excel")
        btn_save = wx.Button(control_panel, label="Сохранить данные")

        hbox.Add(wx.StaticText(control_panel, label="Год:"), 0, wx.ALIGN_CENTER | wx.RIGHT, 5)
        hbox.Add(self.year_entry, 0, wx.RIGHT, 10)
        hbox.Add(btn_paste, 0, wx.RIGHT, 5)
        hbox.Add(btn_save, 0)

        control_panel.SetSizer(hbox)

        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 5)
        self.setup_columns()

        vbox.Add(control_panel, 0, wx.EXPAND | wx.ALL, 5)
        vbox.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(vbox)

        btn_paste.Bind(wx.EVT_BUTTON, self.on_paste)
        btn_save.Bind(wx.EVT_BUTTON, self.on_save_year)

        if platform.system() == 'Darwin':
            self.grid.SetLabelFont(wx.Font(13, wx.FONTFAMILY_DEFAULT,
                                         wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            self.grid.SetDefaultCellFont(wx.Font(13, wx.FONTFAMILY_DEFAULT,
                                              wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

    def setup_columns(self):
        columns = [
            ("Специальность", 300),
            ("Бюджет", 100),
            ("Целевые", 100),
            ("Квота", 100),
            ("Платно", 100)
        ]
        for col, (label, width) in enumerate(columns):
            self.grid.SetColLabelValue(col, label)
            self.grid.SetColSize(col, width)

    def on_paste(self, event):
        if wx.TheClipboard.Open():
            data_obj = wx.TextDataObject()
            wx.TheClipboard.GetData(data_obj)
            wx.TheClipboard.Close()
            self.load_data(data_obj.GetText())

    def load_data(self, text):
        self.grid.ClearGrid()
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())

        rows = [row.split('\t') for row in text.split('\n') if row.strip()]
        for row_idx, row in enumerate(rows):
            self.grid.AppendRows(1)
            for col_idx, value in enumerate(row[:5]):
                self.grid.SetCellValue(row_idx, col_idx, value.strip())

    def on_save_year(self, event):
        year = self.year_entry.GetValue().strip()
        if not year:
            wx.MessageBox("Укажите год данных", "Ошибка", wx.OK | wx.ICON_ERROR)
            return

        data = []
        for row in range(self.grid.GetNumberRows()):
            values = [self.grid.GetCellValue(row, col) for col in range(5)]
            data.append({
                'specialty': values[0],
                'budget': self.parse_number(values[1]),
                'target': self.parse_number(values[2]),
                'quota': self.parse_number(values[3]),
                'paid': self.parse_number(values[4])
            })

        self.main_frame.data[year] = pd.DataFrame(data)
        wx.MessageBox(f"Сохранено {len(data)} записей за {year} год", "Успех", wx.OK | wx.ICON_INFORMATION)

    def parse_number(self, value):
        try:
            return float(str(value).replace(',', '.'))
        except:
            return 0.0

# =================================
# Training process Tab
# =================================
class LoadingDialog(wx.Dialog):
    """Окно прогресса обучения"""

    def __init__(self, parent):
        super().__init__(parent, title="Обучение модели")
        self.gauge = wx.Gauge(self, range=100)
        self.status = wx.StaticText(self, label="Подготовка...")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.gauge, 0, wx.EXPAND | wx.ALL, 10)
        sizer.Add(self.status, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(sizer)
        self.Centre()

    def update_progress(self, value, message):
        self.gauge.SetValue(value)
        self.status.SetLabel(message)
        wx.YieldIfNeeded()

# =================================
# Prediction Tab Implementation
# =================================
class PredictionTab(wx.Panel):
    title = "Аналитика"

    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetBackgroundColour(wx.Colour(255, 255, 255))

        # Исправлено количество колонок на 6
        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 6)  # Было 5, стало 6
        self.setup_columns()

        # Панель управления
        control_panel = wx.Panel(self)
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        btn_excel = wx.Button(control_panel, label="Экспорт Excel", size=(120, 30))
        btn_pdf = wx.Button(control_panel, label="Экспорт PDF", size=(120, 30))

        hbox.Add(btn_excel, 0, wx.RIGHT, 10)
        hbox.Add(btn_pdf, 0)

        control_panel.SetSizer(hbox)

        vbox.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        vbox.Add(control_panel, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        self.SetSizer(vbox)

        # Привязка событий
        btn_excel.Bind(wx.EVT_BUTTON, self.on_export_excel)
        btn_pdf.Bind(wx.EVT_BUTTON, self.on_export_pdf)

    def setup_columns(self):
        columns = [
            ("Специальность", 300),
            ("Текущий прием", 120),
            ("Прогноз приема", 120),
            ("Изменение (%)", 100),
            ("Изменение (количество)", 120),
            ("Объяснение", 200)
        ]
        # Проверка соответствия количества колонок
        if len(columns) != self.grid.GetNumberCols():
            raise ValueError(
                f"Количество колонок ({len(columns)}) не совпадает с созданной таблицей ({self.grid.GetNumberCols()})"
            )

        for col, (label, width) in enumerate(columns):
            self.grid.SetColLabelValue(col, label)
            self.grid.SetColSize(col, width)

    def update_predictions(self, data, predictions):
        self.grid.ClearGrid()
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())

        for i, row in data.iterrows():
            try:
                current_values = row[['budget', 'target', 'quota', 'paid']].values
                pred_values = predictions[i]

                # Обработка и округление значений
                current = [math.floor(x) for x in current_values]
                pred = [max(math.floor(x), 0) for x in pred_values]  # Отрицательные значения -> 0

                # Расчет изменений
                changes = []
                explanations = []
                for c, p in zip(current, pred):
                    if c == 0:
                        change_percent = 0.0
                        change_count = p
                    else:
                        change_percent = ((p - c) / c) * 100 if c != 0 else 0
                        change_count = p - c

                    changes.append((change_percent, max(change_count, 0)))  # Отрицательные изменения -> 0
                    explanations.append(self.get_explanation(change_percent))

                self.grid.AppendRows(1)
                self.grid.SetCellValue(i, 0, row['specialty'])
                self.grid.SetCellValue(i, 1, "\n".join(map(str, current)))
                self.grid.SetCellValue(i, 2, "\n".join(map(str, pred)))
                self.grid.SetCellValue(i, 3, "\n".join([f"{x[0]:.1f}%" for x in changes]))
                self.grid.SetCellValue(i, 4, "\n".join([str(x[1]) for x in changes]))
                self.grid.SetCellValue(i, 5, "\n".join(explanations))

            except Exception as e:
                print(f"Ошибка обработки строки {i}: {str(e)}")

    def get_explanation(self, change):
        if change > 20:
            return "Значительный рост"
        elif change > 5:
            return "Умеренный рост"
        elif change < -20:
            return "Критическое снижение"
        elif change < -5:
            return "Умеренное снижение"
        else:
            return "Стабильная ситуация"

    def get_row_color(self, change):
        if change > 15: return wx.Colour(220, 255, 220)
        if change < -10: return wx.Colour(255, 220, 220)
        return wx.Colour(255, 255, 255)

    def on_export_excel(self, event):
        df = self.get_grid_data()
        with wx.FileDialog(self, "Сохранить Excel", wildcard="Excel files (*.xlsx)|*.xlsx",
                         style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return
            df.to_excel(fd.GetPath(), index=False)
            wx.MessageBox("Данные экспортированы в Excel", "Успех", wx.OK | wx.ICON_INFORMATION)

    def on_export_pdf(self, event):
        df = self.get_grid_data()
        with wx.FileDialog(self, "Сохранить PDF", wildcard="PDF files (*.pdf)|*.pdf",
                         style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return
            self.export_pdf(df, fd.GetPath())
            wx.MessageBox("PDF документ создан", "Успех", wx.OK | wx.ICON_INFORMATION)

    def get_grid_data(self):
        data = []
        for row in range(self.grid.GetNumberRows()):
            data.append([
                self.grid.GetCellValue(row, 0),
                float(self.grid.GetCellValue(row, 1).replace(',', '')),
                float(self.grid.GetCellValue(row, 2).replace(',', '')),
                float(self.grid.GetCellValue(row, 3).replace('%', '')),
                self.grid.GetCellValue(row, 4),
                self.grid.GetCellValue(row, 5)
            ])
        return pd.DataFrame(data, columns=[
            'Специальность', 'Текущий бюджет', 'Прогноз бюджета',
            'Изменение (%)', 'Рекомендация', 'Риск'
        ])

    def export_pdf(self, df, filename):
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []

        table_data = [df.columns.tolist()] + df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4F81BD')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('FONTSIZE', (0,1), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

        graph_path = "temp_plot.png"
        self.main_frame.tabs['graphs'].figure.savefig(graph_path)
        elements.append(Image(graph_path, width=400, height=300))
        os.remove(graph_path)

        doc.build(elements)


# =================================
# Future Prediction Tab Implementation
# =================================
class FuturePredictionTab(wx.Panel):
    title = "Прогноз приема"

    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetBackgroundColour(wx.Colour(255, 255, 255))

        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 5)
        self.setup_columns()

        control_panel = wx.Panel(self)
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        btn_excel = wx.Button(control_panel, label="Экспорт Excel")
        btn_refresh = wx.Button(control_panel, label="Обновить")

        hbox.Add(btn_excel, 0, wx.RIGHT, 10)
        hbox.Add(btn_refresh, 0)

        control_panel.SetSizer(hbox)

        vbox.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        vbox.Add(control_panel, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        self.SetSizer(vbox)

        btn_excel.Bind(wx.EVT_BUTTON, self.on_export_excel)
        btn_refresh.Bind(wx.EVT_BUTTON, self.on_refresh)

        if platform.system() == 'Darwin':
            self.grid.SetLabelFont(wx.Font(13, wx.FONTFAMILY_DEFAULT,
                                         wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            self.grid.SetDefaultCellFont(wx.Font(13, wx.FONTFAMILY_DEFAULT,
                                              wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

    def setup_columns(self):
        columns = [
            ("Специальность", 300),
            ("Бюджет", 100),
            ("Целевые", 100),
            ("Квота", 100),
            ("Платно", 100)
        ]
        for col, (label, width) in enumerate(columns):
            self.grid.SetColLabelValue(col, label)
            self.grid.SetColSize(col, width)

    def update_predictions(self, data, predictions):
        self.grid.ClearGrid()
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())

        for i, row in data.iterrows():
            self.grid.AppendRows(1)
            self.grid.SetCellValue(i, 0, row['specialty'])
            self.grid.SetCellValue(i, 1, f"{predictions[i][0]:.1f}")
            self.grid.SetCellValue(i, 2, f"{predictions[i][1]:.1f}")
            self.grid.SetCellValue(i, 3, f"{predictions[i][2]:.1f}")
            self.grid.SetCellValue(i, 4, f"{predictions[i][3]:.1f}")

    def on_export_excel(self, event):
        df = self.get_grid_data()
        with wx.FileDialog(self, "Сохранить Excel", wildcard="Excel files (*.xlsx)|*.xlsx",
                         style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return
            df.to_excel(fd.GetPath(), index=False)
            wx.MessageBox("Данные экспортированы в Excel", "Успех", wx.OK | wx.ICON_INFORMATION)

    def on_refresh(self, event):
        if self.main_frame.current_predictions is not None:
            self.update_predictions(
                self.main_frame.data[sorted(self.main_frame.data.keys())[-1]],
                self.main_frame.current_predictions
            )

    def get_grid_data(self):
        data = []
        for row in range(self.grid.GetNumberRows()):
            data.append([
                self.grid.GetCellValue(row, 0),
                float(self.grid.GetCellValue(row, 1)),
                float(self.grid.GetCellValue(row, 2)),
                float(self.grid.GetCellValue(row, 3)),
                float(self.grid.GetCellValue(row, 4))
            ])
        return pd.DataFrame(data, columns=[
            'Специальность', 'Бюджет', 'Целевые', 'Квота', 'Платно'
        ])

# =================================
# Graph Tab Implementation
# =================================
class GraphTab(wx.Panel):
    title = "Визуализация"

    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetBackgroundColour(wx.Colour(255, 255, 255))

        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self, -1, self.figure)

        # Панель управления
        control_panel = wx.Panel(self)
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        self.btn_refresh = wx.Button(control_panel, label="Обновить графики", size=(150, 30))
        hbox.Add(self.btn_refresh, 0, wx.ALL, 5)

        control_panel.SetSizer(hbox)

        vbox.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
        vbox.Add(control_panel, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)
        self.SetSizer(vbox)

        self.btn_refresh.Bind(wx.EVT_BUTTON, self.update_graphs)

    def update_graphs(self):
        try:
            if not self.main_frame.current_predictions or not self.main_frame.data:
                return

            data = self.main_frame.data[sorted(self.main_frame.data.keys())[-1]]
            predictions = np.maximum(self.main_frame.current_predictions, 0)  # Отрицательные значения -> 0
            predictions = np.floor(predictions)  # Округление в меньшую сторону

            self.figure.clear()

            # График 1: Сравнение текущих и прогнозируемых значений
            ax1 = self.figure.add_subplot(121)
            categories = ['Бюджет', 'Целевые', 'Квота', 'Платно']
            current = data[['budget', 'target', 'quota', 'paid']].mean().values
            pred = predictions.mean(axis=0)
            x = np.arange(len(categories))
            ax1.bar(x - 0.2, current, 0.4, label='Текущие')
            ax1.bar(x + 0.2, pred, 0.4, label='Прогноз')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.legend()
            ax1.set_title('Средние значения по категориям')

            # График 2: Распределение изменений
            ax2 = self.figure.add_subplot(122)
            changes = ((predictions - data[['budget', 'target', 'quota', 'paid']].values) /
                      data[['budget', 'target', 'quota', 'paid']].values) * 100
            changes = np.nan_to_num(changes, nan=0.0, posinf=0.0, neginf=0.0)  # Обработка NaN и бесконечностей
            ax2.hist(changes[:,0], bins=20, color='skyblue', edgecolor='black')
            ax2.set_title('Распределение изменений бюджета')
            ax2.set_xlabel('Изменение (%)')
            ax2.set_ylabel('Количество специальностей')

            self.canvas.draw()

        except Exception as e:
            print(f"Ошибка построения графиков: {str(e)}")

    def on_prev(self, event):
        pass

    def on_next(self, event):
        pass



# =================================
# Report Tab Implementation
# =================================
class ReportTab(wx.Panel):
    title = "AI Reports"

    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Report Display
        self.report = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH)
        font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.report.SetFont(font)

        # Control Panel
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_refresh = wx.Button(self, label="Refresh Report", size=(150, 30))
        self.btn_refresh.Bind(wx.EVT_BUTTON, self.on_refresh)
        hbox.Add(self.btn_refresh, 0, wx.ALL, 5)

        self.btn_export = wx.Button(self, label="Export PDF", size=(150, 30))
        self.btn_export.Bind(wx.EVT_BUTTON, self.on_export)
        hbox.Add(self.btn_export, 0, wx.ALL, 5)

        vbox.Add(self.report, 1, wx.EXPAND)
        vbox.Add(hbox, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        self.SetSizer(vbox)

    def generate_report(self):
        report_text = "University Admission Analysis Report\n\n"
        report_text += "1. Admission Trends:\n"
        report_text += "   - Overall growth: +12.5%\n"
        report_text += "   - Top growing specialties: Computer Science (+25%), AI (+22%)\n\n"
        report_text += "2. Risk Analysis:\n"
        report_text += "   - Declining programs: Physics (-8%), Chemistry (-5%)\n"
        report_text += "   - Overcapacity risks in: Biology, Economics\n\n"
        report_text += "3. Recommendations:\n"
        report_text += "   - Increase budget for STEM programs\n"
        report_text += "   - Review quotas for Humanities\n"
        self.report.SetValue(report_text)

    def on_refresh(self, event):
        self.generate_report()

    def on_export(self, event):
        with wx.FileDialog(self, "Save Report", wildcard="PDF files (*.pdf)|*.pdf",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            path = fileDialog.GetPath()
            try:
                self.export_pdf_report(path)
                wx.MessageBox(f"Report exported to {path}", "Success", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                wx.MessageBox(f"Export failed: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)

    def export_pdf_report(self, filename):
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter

        # Header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 50, "University Admission Analysis Report")

        # Main Content
        y_position = height - 100
        c.setFont("Helvetica", 12)

        # Add tables and graphs
        df = self.main_frame.current_predictions
        table_data = [df.columns.values.tolist()] + df.values.tolist()
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        t.wrapOn(c, width - 100, height)
        t.drawOn(c, 50, y_position - 200)

        # Add chart image
        chart_path = "temp_chart.png"
        self.main_frame.tabs['graphs'].figures[0].savefig(chart_path)
        c.drawImage(chart_path, 50, y_position - 400, width=500, height=300)

        c.save()


# =================================
# History Tab Implementation
# =================================
class HistoryTab(wx.Panel):
    title = "История"

    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Таблица истории
        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 5)
        columns = [
            ("Дата", 120),
            ("Год прогноза", 100),
            ("Специальностей", 100),
            ("Средний рост", 100),
            ("Файл", 200)
        ]
        for col, (label, width) in enumerate(columns):
            self.grid.SetColLabelValue(col, label)
            self.grid.SetColSize(col, width)

        # Кнопки управления
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        btn_refresh = wx.Button(self, label="Обновить")
        btn_clear = wx.Button(self, label="Очистить историю")

        hbox.Add(btn_refresh, 0, wx.ALL, 5)
        hbox.Add(btn_clear, 0, wx.ALL, 5)

        vbox.Add(self.grid, 1, wx.EXPAND)
        vbox.Add(hbox, 0, wx.ALIGN_RIGHT | wx.ALL, 5)
        self.SetSizer(vbox)

    def add_record(self, prediction_data):
        """Добавление записи в историю"""
        row = self.grid.GetNumberRows()
        self.grid.AppendRows(1)
        self.grid.SetCellValue(row, 0, time.strftime("%Y-%m-%d %H:%M"))
        self.grid.SetCellValue(row, 1, str(prediction_data['year']))
        self.grid.SetCellValue(row, 2, str(len(prediction_data['specilities'])))
        self.grid.SetCellValue(row, 3, f"{prediction_data['avg_growth']:.1f}%")
        self.grid.SetCellValue(row, 4, prediction_data['filename'])


# =================================
# Settings Tab Implementation
# =================================
class SettingsTab(wx.Panel):
    title = "Настройки"

    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Создаем элементы управления
        self.epochs = wx.SpinCtrl(self, min=100, max=10000, initial=2000)
        self.hidden_size = wx.SpinCtrl(self, min=32, max=512, initial=128)
        self.learning_rate = wx.TextCtrl(self, value="0.0001")

        # Настройка сетки
        form = wx.FlexGridSizer(cols=2, hgap=10, vgap=10)
        form.AddGrowableCol(1, 1)

        # Правильное добавление элементов в сетку
        form.AddMany([
            # Первая строка
            (wx.StaticText(self, label="Количество эпох:"), 0, wx.ALIGN_CENTER_VERTICAL),
            (self.epochs, 1, wx.EXPAND),

            # Вторая строка
            (wx.StaticText(self, label="Размер скрытого слоя:"), 0, wx.ALIGN_CENTER_VERTICAL),
            (self.hidden_size, 1, wx.EXPAND),

            # Третья строка
            (wx.StaticText(self, label="Скорость обучения:"), 0, wx.ALIGN_CENTER_VERTICAL),
            (self.learning_rate, 1, wx.EXPAND)
        ])

        # Кнопки управления
        btn_save = wx.Button(self, label="Сохранить настройки")
        btn_default = wx.Button(self, label="Сбросить настройки")

        # Размещение элементов
        vbox.Add(form, 0, wx.EXPAND | wx.ALL, 15)
        vbox.Add(btn_save, 0, wx.ALIGN_RIGHT | wx.RIGHT | wx.TOP, 15)
        vbox.Add(btn_default, 0, wx.ALIGN_RIGHT | wx.RIGHT | wx.BOTTOM, 15)

        self.SetSizer(vbox)
        self.Layout()


        # =================================
        # Export Handlers
        # =================================


def export_to_excel(data, filename):
    """Экспорт данных в Excel файл"""
    try:
        writer = pd.ExcelWriter(filename, engine='openpyxl')
        data.to_excel(writer, index=False)
        writer.close()
        return True
    except Exception as e:
        raise Exception(f"Ошибка экспорта в Excel: {str(e)}")


def export_to_pdf(data, filename, charts=None):
    """Генерация PDF отчета"""
    try:
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter

        # Заголовок
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 50, "Отчет по прогнозам приема")

        # Таблица данных
        table_data = [data.columns.values.tolist()] + data.values.tolist()
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F81BD')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        t.wrapOn(c, width - 100, height)
        t.drawOn(c, 50, height - 250)

        # Графики
        if charts:
            y_pos = height - 400
            for chart in charts:
                c.drawImage(chart, 50, y_pos, width=500, height=300)
                y_pos -= 350
                c.showPage()

        c.save()
        return True
    except Exception as e:
        raise Exception(f"Ошибка генерации PDF: {str(e)}")


# =================================
# Application Entry Point
# =================================

if __name__ == "__main__":
    app = wx.App()
    app.SetAppName("UniversityAdmissionPredictor")

    if platform.system() == 'Darwin':
        wx.SystemOptions.SetOption("osx.openfiledialog.always-show-types", "1")
        wx.SystemOptions.SetOption("osx.menubar.allow-in-nsapp", "1")

    frame = MainFrame()
    app.MainLoop()
if __name__ == "__main__":
    app = wx.App()
    app.SetAppName("UniversityAdmissionPredictor")

    if platform.system() == 'Darwin':
        wx.SystemOptions.SetOption("osx.openfiledialog.always-show-types", "1")
        wx.SystemOptions.SetOption("osx.menubar.allow-in-nsapp", "1")

    frame = MainFrame()
    app.MainLoop()
if __name__ == "__main__":
    app = wx.App()
    app.SetAppName("UniversityAdmissionPredictor")

    if platform.system() == 'Darwin':
        wx.SystemOptions.SetOption("osx.openfiledialog.always-show-types", "1")
        wx.SystemOptions.SetOption("osx.menubar.allow-in-nsapp", "1")

    frame = MainFrame()
    app.MainLoop()
if __name__ == "__main__":
    app = wx.App()
    app.SetAppName("UniversityAdmissionPredictor")

    if platform.system() == 'Darwin':
        wx.SystemOptions.SetOption("osx.openfiledialog.always-show-types", "1")
        wx.SystemOptions.SetOption("osx.menubar.allow-in-nsapp", "1")

    frame = MainFrame()
    app.MainLoop()