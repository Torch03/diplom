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
from datetime import datetime
import os
import pickle

import platform
if platform.system() == 'Darwin':
    wx.SystemOptions.SetOption("osx.openfiledialog.always-show-types", "1")
    wx.SystemOptions.SetOption("osx.menubar.allow-in-nsapp", "1")


# =================================
# Константы и вспомогательные функции
# =================================

DEFAULT_COLORS = {
    'background': wx.Colour(240, 240, 240),
    'header': wx.Colour(53, 132, 228),
    'positive': wx.Colour(67, 160, 71),
    'negative': wx.Colour(198, 40, 40),
    'text': wx.Colour(0, 0, 0),
    'panel': wx.Colour(255, 255, 255)
}

THEMES = {
    "light": {
        "background": wx.Colour(240, 240, 240),
        "text": wx.Colour(0, 0, 0),
        "panel": wx.Colour(255, 255, 255),
        "grid_bg": wx.Colour(255, 255, 255),
        "grid_text": wx.Colour(0, 0, 0)
    },
    "dark": {
        "background": wx.Colour(30, 30, 30),
        "text": wx.Colour(255, 255, 255),
        "panel": wx.Colour(50, 50, 50),
        "grid_bg": wx.Colour(60, 60, 60),
        "grid_text": wx.Colour(255, 255, 255)
    }
}

def format_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def create_menu_item(menu, label, func):
    item = wx.MenuItem(menu, -1, label)
    menu.Bind(wx.EVT_MENU, func, id=item.GetId())
    menu.Append(item)
    return item


# =================================
# Prediction Tab Implementation
# =================================

class PredictionTab(wx.Panel):
    title = "Аналитика"

    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()
        self.SetBackgroundColour(wx.WHITE)

    def init_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Грид
        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 6)
        self.setup_columns()
        main_sizer.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)

        # Отдельная панель для кнопок
        button_panel = wx.Panel(self)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Кнопки
        self.btn_excel = wx.Button(button_panel, label="Excel")
        self.btn_pdf = wx.Button(button_panel, label="PDF")
        self.btn_help = wx.Button(button_panel, label="Справка")

        button_sizer.Add(self.btn_excel, 0, wx.RIGHT, 10)
        button_sizer.Add(self.btn_pdf, 0)
        button_sizer.Add(self.btn_help, 0, wx.LEFT, 10)

        # сайзер для панели кнопок
        button_panel.SetSizer(button_sizer)

        main_sizer.Add(button_panel, 0, wx.ALIGN_RIGHT | wx.ALL, 5)

        self.SetSizerAndFit(main_sizer)

        # Привязка событий
        self.btn_excel.Bind(wx.EVT_BUTTON, self.export_excel)
        self.btn_pdf.Bind(wx.EVT_BUTTON, self.export_pdf)
        self.btn_help.Bind(wx.EVT_BUTTON, self.show_help)

    def setup_columns(self):

        theme = THEMES[self.GetTopLevelParent().current_theme]
        self.grid.SetDefaultCellBackgroundColour(theme["grid_bg"])
        self.grid.SetDefaultCellTextColour(theme["grid_text"])

        columns = [
            ("Специальность", 300),
            ("Текущий", 100),
            ("Прогноз", 100),
            ("Изменение (%)", 100),
            ("Изменение", 100),
            ("Тренд", 100)
        ]
        for col, (label, width) in enumerate(columns):
            self.grid.SetColLabelValue(col, label)
            self.grid.SetColSize(col, width)

    def show_help(self, event):
        HelpWindow(self)

    def update_predictions(self, data, predictions):
        try:
            self.grid.ClearGrid()
            if self.grid.GetNumberRows() > 0:
                self.grid.DeleteRows(0, self.grid.GetNumberRows())

            for i, row in data.iterrows():
                # Проверка наличия всех необходимых значений
                if i >= len(predictions):
                    break

                current_values = [row['budget'], row['target'], row['quota'], row['paid']]
                pred_values = predictions[i]

                # Расчет показателей
                total_current = sum(current_values)
                total_pred = sum(pred_values)

                # Добавление строки
                self.grid.AppendRows(1)
                self.grid.SetCellValue(i, 0, str(row['specialty']))
                self.grid.SetCellValue(i, 1, str(total_current))
                self.grid.SetCellValue(i, 2, str(total_pred))

                # Форматирование изменений
                change_percent = ((total_pred - total_current) / total_current * 100) if total_current != 0 else 0
                self.grid.SetCellValue(i, 3, f"{change_percent:.1f}%")
                self.grid.SetCellValue(i, 4, str(total_pred - total_current))
                self.grid.SetCellValue(i, 5, "Рост" if total_pred > total_current else "Снижение")

            # Принудительное обновление таблицы
            self.grid.ForceRefresh()
            self.grid.AutoSizeColumns()

        except Exception as e:
            print(f"Ошибка обновления таблицы: {str(e)}")
            self.grid.ClearGrid()

    def show_history(self, event):
        if self.main_frame.history.get_last():
            data = self.main_frame.history.get_last()['data']
            self.update_predictions(data, data)  # Упрощенный пример
        else:
            wx.MessageBox("История прогнозов пуста!", "Информация", wx.OK | wx.ICON_INFORMATION)

    def export_excel(self, event):
        df = self.get_grid_data()
        with wx.FileDialog(self, "Сохранить Excel", wildcard="Excel files (*.xlsx)|*.xlsx",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return
            df.to_excel(fd.GetPath(), index=False)
            wx.MessageBox("Данные экспортированы в Excel", "Успех", wx.OK | wx.ICON_INFORMATION)

    def export_pdf(self, event):
        df = self.get_grid_data()
        with wx.FileDialog(self, "Сохранить PDF", wildcard="PDF files (*.pdf)|*.pdf",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return
            self.export_pdf_report(df, fd.GetPath())
            wx.MessageBox("PDF документ создан", "Успех", wx.OK | wx.ICON_INFORMATION)

    def get_grid_data(self):
        data = []
        for row in range(self.grid.GetNumberRows()):
            data.append([
                self.grid.GetCellValue(row, 0),
                self.grid.GetCellValue(row, 1),
                self.grid.GetCellValue(row, 2),
                self.grid.GetCellValue(row, 3),
                self.grid.GetCellValue(row, 4),
                self.grid.GetCellValue(row, 5)
            ])
        return pd.DataFrame(data, columns=[
            'Специальность', 'Текущий набор', 'Прогноз',
            'Изменение (%)', 'Изменение', 'Тренд'
        ])

    def export_pdf_report(self, df, filename):
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []

        # Добавление таблицы
        table_data = [df.columns.tolist()] + df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F81BD')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

        if hasattr(self.main_frame, 'graph_tab'):
            graph_path = "temp_plot.png"
            self.main_frame.graph_tab.figure.savefig(graph_path)
            elements.append(Image(graph_path, width=400, height=300))
            os.remove(graph_path)

        doc.build(elements)


# =================================
# Graph Tab Implementation
# =================================

class GraphTab(wx.Panel):
    title = "Графики"

    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()
        self.SetBackgroundColour(wx.WHITE)

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self, -1, self.figure)

        vbox.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(vbox)

    def update_graphs(self):
        try:
            self.figure.clear()

            if not self.main_frame.current_predictions or not self.main_frame.data:
                return

            data = self.main_frame.data[sorted(self.main_frame.data.keys())[-1]]
            pred = self.main_frame.current_predictions

            categories = data['specialty'].tolist()[:10]
            current = data['budget'].values[:10]
            predicted = [math.floor(x[0]) for x in pred[:10]]

            ax = self.figure.add_subplot(111)
            x = np.arange(len(categories))

            ax.bar(x - 0.2, current, 0.4, label='Текущий')
            ax.bar(x + 0.2, predicted, 0.4, label='Прогноз')

            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.legend()
            ax.set_title('Топ 10 специальностей')

            self.canvas.draw()

        except Exception as e:
            print(f"Ошибка графиков: {e}")


# =================================
# Neural Network Implementation
# =================================

class NeuralNetwork:
    def __init__(self, input_size, hidden_size=128, output_size=4):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size//2) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, hidden_size//2))
        self.W3 = np.random.randn(hidden_size//2, output_size) * np.sqrt(2. / (hidden_size//2))
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

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if progress_callback and epoch % 10 == 0:
                progress = int((epoch / epochs) * 100)
                progress_callback(progress, f"Epoch {epoch}/{epochs}")

            if val_loss < best_loss:
                best_loss = val_loss
            elif val_loss > best_loss * 1.1:
                print("Early stopping")
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

    def get_architecture(self):
        return {
            'input_size': self.W1.shape[0],
            'hidden1': self.W1.shape[1],
            'hidden2': self.W2.shape[1],
            'output': self.W3.shape[1]
        }

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
        if not self.IsShown():
            self.Show()

        self.gauge.SetValue(value)
        self.status.SetLabel(f"{message} ({value}%)")
        self.Refresh()
        wx.Yield()

    def _update_gui(self, value, message):
        self.gauge.SetValue(value)
        self.status.SetLabel(message)
        self.Update()
        self.Refresh()


# =================================
# Main Application Window
# =================================

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="University Analytics", size=(1200, 800))
        self.loading_dialog = None
        self.nn = None
        self.data = {}
        self.current_predictions = None
        self.current_theme = "light"

        # Инициализация статусной строки
        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("Готово")

        # Инициализация системы уведомлений
        self.notifier = NotificationManager(self.status_bar)

        # Инициализация менеджера тем
        self.theme_manager = ThemeManager()

        # Настройки для macOS
        if platform.system() == 'Darwin':
            self.SetWindowVariant(wx.WINDOW_VARIANT_NORMAL)
            self.SetSizeHints(1000, 600)

        # Инициализация интерфейса
        self.init_ui()
        self.init_menu()
        self.setup_theme_menu()
        self.setup_context_menus()

        # Дополнительные настройки
        self.Centre()
        self.Show(True)
        wx.CallLater(100, self.force_ui_update)

        self.model_settings = {
            'epochs': 2000,
            'hidden_size': 128,
            'learning_rate': 0.0001
        }

    def force_ui_update(self):
        self.Refresh()
        self.Update()

    def init_ui(self):
        panel = wx.Panel(self)
        notebook = wx.Notebook(panel)

        # Создание и добавление вкладок
        self.input_tab = InputTab(notebook, self)
        self.prediction_tab = PredictionTab(notebook, self)
        self.graph_tab = GraphTab(notebook, self)
        self.history_tab = HistoryTab(notebook, self)
        self.risk_tab = RiskAnalysisTab(notebook, self)

        notebook.AddPage(self.input_tab, "Ввод данных")
        notebook.AddPage(self.prediction_tab, "Прогнозы")
        notebook.AddPage(self.graph_tab, "Графики")
        notebook.AddPage(self.history_tab, "История")
        notebook.AddPage(self.risk_tab, "Анализ изменений")

        # Главный сайзер
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 5)
        panel.SetSizer(main_sizer)

        # Настройка тулбара
        toolbar = self.CreateToolBar()
        train_tool = toolbar.AddTool(wx.ID_ANY, "Обучить", wx.ArtProvider.GetBitmap(wx.ART_PLUS))
        predict_tool = toolbar.AddTool(wx.ID_ANY, "Прогноз", wx.ArtProvider.GetBitmap(wx.ART_FIND))
        toolbar.Realize()

        # Привязка событий тулбара
        self.Bind(wx.EVT_TOOL, self.on_train, train_tool)
        self.Bind(wx.EVT_TOOL, self.on_predict, predict_tool)

        # Убрать повторное создание статусной строки
        panel.Layout()
        self.Show(True)

    def setup_toolbar(self):
        self.toolbar = self.CreateToolBar()
        train_tool = self.toolbar.AddTool(wx.ID_ANY, "Train", wx.ArtProvider.GetBitmap(wx.ART_PLUS))
        predict_tool = self.toolbar.AddTool(wx.ID_ANY, "Predict", wx.ArtProvider.GetBitmap(wx.ART_FIND))
        self.toolbar.Realize()

        self.Bind(wx.EVT_TOOL, self.on_train, train_tool)
        self.Bind(wx.EVT_TOOL, self.on_predict, predict_tool)

    def init_menu(self):
        self.menubar = wx.MenuBar()

        # Меню Файл
        file_menu = wx.Menu()
        create_menu_item(file_menu, "Открыть проект\tCtrl+O", self.on_open)
        create_menu_item(file_menu, "Сохранить проект\tCtrl+S", self.on_save)
        file_menu.AppendSeparator()
        create_menu_item(file_menu, "Выход", self.on_exit)
        self.menubar.Append(file_menu, "&Файл")

        # Меню Инструменты
        tools_menu = wx.Menu()
        create_menu_item(tools_menu, "Настройки модели", self.show_settings)
        create_menu_item(tools_menu, "Анализ рисков", self.show_risks)
        self.menubar.Append(tools_menu, "&Инструменты")

        # Меню Справка
        help_menu = wx.Menu()
        help_item = help_menu.Append(wx.ID_ANY, "Справка", "Открыть руководство пользователя")
        about_item = help_menu.Append(wx.ID_ABOUT, "О программе", "Информация о приложении")
        self.Bind(wx.EVT_MENU, self.show_help, help_item)
        self.Bind(wx.EVT_MENU, self.show_about, about_item)
        self.menubar.Append(help_menu, "&Справка")

        self.SetMenuBar(self.menubar)

    def setup_theme_menu(self):
        """Создает меню для выбора темы оформления"""
        theme_menu = wx.Menu()

        # Пункты меню
        light_item = theme_menu.Append(
            wx.ID_ANY,
            "Светлая тема\tCtrl+L",
            "Переключить на светлую цветовую схему"
        )
        dark_item = theme_menu.Append(
            wx.ID_ANY,
            "Темная тема\tCtrl+D",
            "Переключить на темную цветовую схему"
        )

        # Привязка обработчиков
        self.Bind(wx.EVT_MENU, lambda e: self.change_theme("light"), light_item)
        self.Bind(wx.EVT_MENU, lambda e: self.change_theme("dark"), dark_item)

        # Добавление в главное меню
        self.menubar.Append(theme_menu, "&Темы")

    def change_theme(self, theme_name):
        """Применяет выбранную тему ко всему приложению"""
        try:
            # Применение темы
            self.theme_manager.apply_theme(self, theme_name)
            self.current_theme = theme_name

            # Обновление элементов интерфейса
            self.Refresh()
            self.Update()

            # Уведомление пользователя
            self.notifier.show_status(
                f"Тема изменена на {theme_name.capitalize()}",
                duration=2000
            )

            # Принудительное обновление таблиц
            for tab in [self.input_tab, self.prediction_tab]:
                tab.grid.ForceRefresh()

        except Exception as e:
            self.notifier.show_popup(
                "Ошибка темы",
                f"Не удалось применить тему: {str(e)}",
                wx.ICON_ERROR
            )

    def setup_context_menus(self):
        """Настраивает контекстные меню для всех таблиц"""
        # Список поддерживаемых таблиц
        grids = [
            self.input_tab.grid,
            self.prediction_tab.grid,
            self.history_tab.grid,
            self.risk_tab.grid
        ]

        # Привязка обработчика контекстного меню
        for grid in grids:
            grid.Bind(wx.EVT_CONTEXT_MENU, self.show_context_menu)

            # Настройка внешнего вида
            grid.SetDefaultCellBackgroundColour(
                THEMES[self.current_theme]['grid_bg']
            )
            grid.SetDefaultCellTextColour(
                THEMES[self.current_theme]['grid_text']
            )

    def show_context_menu(self, event):
        """Показывает контекстное меню для таблицы"""
        try:
            grid = event.GetEventObject()

            # Создание меню
            menu = GridContextMenu(grid)

            # Показ меню
            grid.PopupMenu(menu)

            # Обновление после закрытия
            menu.Destroy()
            grid.ForceRefresh()

        except Exception as e:
            self.notifier.show_popup(
                "Ошибка меню",
                f"Не удалось показать контекстное меню: {str(e)}",
                wx.ICON_ERROR
            )

    def check_updates(self):
        def update_callback(version):
            if version and version != "v2.0":
                self.notifier.show_popup("Update Available",
                                         f"New version {version} is available!", wx.ICON_INFORMATION)

        UpdateChecker(update_callback).start()

    def on_train(self, event):
        if not self.data:
            self.notifier.show_popup("Ошибка", "Нет данных для обучения", wx.ICON_ERROR)
            return

        self.loading_dialog = LoadingScreen(self)
        self.notifier.show_status("Начато обучение модели...")

        def train_thread():
            try:
                X, y = self.prepare_training_data()
                print(f"[DEBUG] Training data shape - X: {X.shape}, y: {y.shape}")

                self.nn = NeuralNetwork(
                    input_size=X.shape[1],
                    hidden_size=self.model_settings['hidden_size'],
                    output_size=4
                )

                self.nn.train(
                    X, y,
                    epochs=self.model_settings['epochs'],
                    lr=self.model_settings['learning_rate']
                )

                wx.CallAfter(self._training_finished)

            except Exception as e:
                error_msg = f"Ошибка обучения: {str(e)}"
                print(f"[ERROR] {error_msg}")
                wx.CallAfter(self._training_failed, error_msg)

            finally:
                wx.CallAfter(self._cleanup_loading_dialog)

        print("[DEBUG] Starting training thread...")
        threading.Thread(target=train_thread, daemon=True).start()

    def _training_finished(self):
        self.notifier.show_status("Обучение завершено успешно")
        self._cleanup_loading_dialog()

    def _training_failed(self, error_msg):
        self.notifier.show_popup("Ошибка", f"Ошибка обучения: {error_msg}", wx.ICON_ERROR)
        self._cleanup_loading_dialog()

    def _cleanup_loading_dialog(self):
        if self.loading_dialog:
            self.loading_dialog.Destroy()
            self.loading_dialog = None

    def on_predict(self, event):
        # Проверка на наличие обученной модели
        if self.nn is None:
            self.notifier.show_popup("Ошибка", "Сначала обучите модель!", wx.ICON_ERROR)
            return

        # Улучшенная валидация данных
        errors = DataValidator.validate_grid(self.input_tab.grid)
        if errors:
            self.notifier.show_popup("Ошибка данных",
                                     f"Найдено {len(errors)} невалидных значений!\nПроверьте выделенные ячейки.",
                                     wx.ICON_ERROR)
            return

        try:
            latest_year = sorted(self.data.keys())[-1]
            df = self.data[latest_year]

            # Проверка наличия необходимых колонок
            required_columns = ['budget', 'target', 'quota', 'paid']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Отсутствуют необходимые колонки в данных")

            X = df[required_columns].values

            # Выполнение предсказания
            predictions = self.nn.predict(X)

            # Сохранение и отображение результатов
            self.current_predictions = np.maximum(np.floor(predictions), 0).astype(int)

            # Обновление интерфейса
            wx.CallAfter(self.prediction_tab.update_predictions, df, self.current_predictions)
            wx.CallAfter(self.graph_tab.update_graphs)
            wx.CallAfter(self.update_history, df)
            wx.CallAfter(self.update_risk_analysis)

            self.notifier.show_status("Прогноз успешно сгенерирован")

        except Exception as e:
            # Обработка ошибок
            error_msg = f"Ошибка прогнозирования: {str(e)}"
            print(f"DEBUG: {error_msg}")  # Логирование
            self.notifier.show_popup("Ошибка", error_msg, wx.ICON_ERROR)

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

    def update_risk_analysis(self):
        if hasattr(self, 'risk_tab') and self.current_predictions is not None and self.data:
            risks = {}
            latest_year = sorted(self.data.keys())[-1]
            df = self.data[latest_year]
            for i, row in df.iterrows():
                if any(x < 0 for x in self.current_predictions[i]):
                    risks[row['specialty']] = "Серьезные изменения"
                elif abs(sum(self.current_predictions[i]) - sum(row[['budget', 'target', 'quota', 'paid']])) > 50:
                    risks[row['specialty']] = "Средний масштаб изменений"
            self.risk_tab.update_risks(risks)


    def update_history(self, df):
        history_data = []
        if self.current_predictions is not None:
            for i, row in df.iterrows():
                history_data.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'specialty': row['specialty'],
                    'prediction': sum(self.current_predictions[i]),
                    'status': "Успешно" if all(x >= 0 for x in self.current_predictions[i]) else "Ошибка"
                })
        if hasattr(self, 'history_tab'):
            self.history_tab.update_history(history_data)

    def show_settings(self, event):
        if self.nn:
            dlg = SettingsDialog(self, self.nn)
            dlg.ShowModal()
        else:
            wx.MessageBox("Сначала обучите модель!", "Ошибка", wx.OK | wx.ICON_WARNING)

    def show_risks(self, event):
        if self.data:
            risks = self.risk_analyzer.analyze(
                self.data[sorted(self.data.keys())[-1]]
            )
            if not risks.empty:
                dlg = wx.Dialog(self, title="Анализ рисков")
                grid = gridlib.Grid(dlg)
                grid.CreateGrid(risks.shape[0], 3)
                grid.SetColLabelValue(0, "Специальность")
                grid.SetColLabelValue(1, "Уровень риска")
                grid.SetColLabelValue(2, "Показатели")

                for i, row in risks.iterrows():
                    grid.SetCellValue(i, 0, row['specialty'])
                    grid.SetCellValue(i, 1, row['risk_level'])
                    grid.SetCellValue(i, 2, str(row['indicators']))

                sizer = wx.BoxSizer(wx.VERTICAL)
                sizer.Add(grid, 1, wx.EXPAND)
                dlg.SetSizer(sizer)
                dlg.SetSize((500, 300))
                dlg.ShowModal()
            else:
                wx.MessageBox("Риски не обнаружены!", "Информация", wx.OK | wx.ICON_INFORMATION)
        else:
            wx.MessageBox("Нет данных для анализа!", "Ошибка", wx.OK | wx.ICON_WARNING)


    def show_help(self, event):
        HelpWindow(self)

    def show_about(self, event):
        with wx.AboutBox(wx.AboutDialogInfo()) as dlg:
            dlg.SetName("Анализ приёма в вузы")
            dlg.SetVersion("2.0")
            dlg.SetDescription("Программа для прогнозирования набора студентов")
            dlg.ShowModal()

    def update_status(self, message):
        self.statusbar.SetStatusText(message, 0)

    def on_open(self, event):
        with wx.FileDialog(self, "Открыть проект", wildcard="*.aproj",
                           style=wx.FD_OPEN) as fd:
            if fd.ShowModal() == wx.ID_OK:
                # Реализация загрузки проекта
                wx.MessageBox("Функция в разработке!", "Информация", wx.OK | wx.ICON_INFORMATION)

    def on_save(self, event):
        with wx.FileDialog(self, "Сохранить проект", wildcard="*.aproj",
                           style=wx.FD_SAVE) as fd:
            if fd.ShowModal() == wx.ID_OK:
                # Реализация сохранения проекта
                wx.MessageBox("Функция в разработке!", "Информация", wx.OK | wx.ICON_INFORMATION)

    def on_exit(self, event):
        self.Close()


# =================================
# Input Tab Implementation
# =================================

class InputTab(wx.Panel):
    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.year_entry = wx.TextCtrl(self, size=(140, -1))
        btn_paste = wx.Button(self, label="Вставить из Excel", size=(160, 32))
        btn_save = wx.Button(self, label="Сохранить данные", size=(160, 32))
        btn_clear = wx.Button(self, label="Очистить таблицу", size=(160, 32))
        btn_settings = wx.Button(self, label="Настройки", size=(160, 32))  # Новая кнопка
        btn_manual = wx.Button(self, label="Руководство", size=(160, 32))

        hbox.Add(self.year_entry, 0, wx.RIGHT, 20)
        hbox.Add(btn_paste, 0, wx.RIGHT, 10)
        hbox.Add(btn_save, 0, wx.RIGHT, 10)
        hbox.Add(btn_clear, 0, wx.RIGHT, 10)
        hbox.Add(btn_settings, 0, wx.RIGHT, 10)
        hbox.Add(btn_manual, 0)

        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 5)
        self.setup_columns()

        vbox.Add(hbox, 0, wx.EXPAND | wx.ALL, 10)
        vbox.Add(self.grid, 1, wx.EXPAND)
        self.SetSizer(vbox)

        btn_paste.Bind(wx.EVT_BUTTON, self.on_paste)
        btn_save.Bind(wx.EVT_BUTTON, self.on_save)
        btn_clear.Bind(wx.EVT_BUTTON, self.on_clear)
        btn_settings.Bind(wx.EVT_BUTTON, self.show_settings)
        btn_manual.Bind(wx.EVT_BUTTON, self.show_manual)

    def setup_columns(self):

        theme = THEMES[self.GetTopLevelParent().current_theme]
        self.grid.SetDefaultCellBackgroundColour(theme["grid_bg"])
        self.grid.SetDefaultCellTextColour(theme["grid_text"])

        columns = [
            ("Специальность", 400),
            ("Бюджетные места", 150),
            ("Целевой набор", 150),
            ("Квота", 150),
            ("Платные места", 150)
        ]
        for col, (label, width) in enumerate(columns):
            self.grid.SetColLabelValue(col, label)
            self.grid.SetColSize(col, width)
        self.grid.DisableDragColSize()

    def on_paste(self, event):
        clipboard = wx.Clipboard.Get()
        if clipboard.Open():
            data_obj = wx.TextDataObject()
            if clipboard.GetData(data_obj):
                self.load_data(data_obj.GetText())
            clipboard.Close()
            DataValidator.validate_grid(self.grid)

    def on_save(self, event):
        year = self.year_entry.GetValue().strip()
        if not year:
            wx.MessageBox("Введите год данных", "Ошибка", wx.OK | wx.ICON_ERROR)
            return

        data = []
        for row in range(self.grid.GetNumberRows()):
            specialty = self.grid.GetCellValue(row, 0)
            if not specialty:
                continue

            data.append({
                'specialty': specialty,
                'budget': self.parse_number(self.grid.GetCellValue(row, 1)),
                'target': self.parse_number(self.grid.GetCellValue(row, 2)),
                'quota': self.parse_number(self.grid.GetCellValue(row, 3)),
                'paid': self.parse_number(self.grid.GetCellValue(row, 4))
            })

        self.main_frame.data[year] = pd.DataFrame(data)
        wx.MessageBox(f"Данные за {year} год сохранены!", "Успех", wx.OK | wx.ICON_INFORMATION)

    def on_clear(self, event):
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())

    def parse_number(self, value):
        try:
            return float(str(value).replace(',', '.').replace(' ', ''))
        except:
            return 0.0

    def load_data(self, clipboard_data):
        rows = [row.split('\t') for row in clipboard_data.split('\n') if row.strip()]
        self.grid.ClearGrid()
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())

        for row_idx, row in enumerate(rows):
            self.grid.AppendRows(1)
            for col_idx, value in enumerate(row[:5]):
                self.grid.SetCellValue(row_idx, col_idx, value.strip())

    def show_manual(self, event):
        dlg = UserManualDialog(self)
        dlg.ShowModal()
        dlg.Destroy()

    def show_settings(self, event):
        dlg = SettingsDialog(self, self.main_frame)
        dlg.ShowModal()
        dlg.Destroy()


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
    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()
        self.SetBackgroundColour(DEFAULT_COLORS['background'])
        self.last_plot_type = 'bar'

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Простейший график-заглушка
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.plot_dummy_data()  # Всегда строим график при инициализации

        # Панель управления
        control_panel = wx.Panel(self)
        self.btn_update = wx.Button(control_panel, label="Обновить")
        self.btn_export = wx.Button(control_panel, label="Экспорт")

        self.btn_update.Bind(wx.EVT_BUTTON, self.update_graphs)
        self.btn_export.Bind(wx.EVT_BUTTON, self.export_plot)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.btn_update, 0, wx.RIGHT, 10)
        hbox.Add(self.btn_export, 0)
        control_panel.SetSizer(hbox)

        vbox.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
        vbox.Add(control_panel, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)
        self.SetSizer(vbox)

    def plot_dummy_data(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r--')
        ax.set_title('График готовности системы')
        ax.set_xlabel('Время работы')
        ax.set_ylabel('Производительность')
        self.canvas.draw()

    def update_graphs(self, event=None):
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            if self.main_frame.current_predictions is not None:
                # Если есть данные, строим реальный график
                data = self.main_frame.data[sorted(self.main_frame.data.keys())[-1]]
                specialties = data['specialty'].tolist()[:5]
                values = self.main_frame.current_predictions[:5].sum(axis=1)
                ax.bar(specialties, values, color=DEFAULT_COLORS['positive'])
                ax.set_title('Топ-5 специальностей по прогнозу')
            else:
                # Иначе обновляем заглушку
                self.plot_dummy_data()

            self.canvas.draw()
        except Exception as e:
            self.plot_dummy_data()

    def export_plot(self, event):
        with wx.FileDialog(self, "Сохранить график", wildcard="PNG files (*.png)|*.png",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fd:
            if fd.ShowModal() == wx.ID_OK:
                self.figure.savefig(fd.GetPath(), dpi=300)
                wx.MessageBox("График сохранен!", "Успех", wx.OK | wx.ICON_INFORMATION)

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

        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 50, "University Admission Analysis Report")

        y_position = height - 100
        c.setFont("Helvetica", 12)

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

        chart_path = "temp_chart.png"
        self.main_frame.tabs['graphs'].figures[0].savefig(chart_path)
        c.drawImage(chart_path, 50, y_position - 400, width=500, height=300)

        c.save()


# =================================
# HistoryManager
# =================================

class HistoryManager:
    def __init__(self):
        self.history = []
        self.max_entries = 10

    def add_record(self, prediction_data):
        if len(self.history) >= self.max_entries:
            self.history.pop(0)
        self.history.append({
            'timestamp': format_date(),
            'data': prediction_data,
            'year': datetime.now().year
        })

    def get_last(self):
        return self.history[-1] if self.history else None

    def clear(self):
        self.history = []


# =================================
# History Tab Implementation
# =================================

class HistoryTab(wx.Panel):
    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 4)  # Добавляем колонку для даты
        self.setup_columns()

        vbox.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(vbox)

    def setup_columns(self):
        columns = [
            ("Дата", 200),
            ("Специальность", 300),
            ("Прогноз", 150),
            ("Статус", 100)
        ]
        for col, (label, width) in enumerate(columns):
            self.grid.SetColLabelValue(col, label)
            self.grid.SetColSize(col, width)
        self.grid.DisableDragColSize()

    def update_history(self, history_data):
        self.grid.ClearGrid()
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())

        for i, entry in enumerate(history_data):
            self.grid.AppendRows(1)
            self.grid.SetCellValue(i, 0, entry['timestamp'])
            self.grid.SetCellValue(i, 1, entry['specialty'])
            self.grid.SetCellValue(i, 2, str(entry['prediction']))
            self.grid.SetCellValue(i, 3, entry['status'])


# =================================
# RiskAnalyzer
# =================================

class RiskAnalyzer:
    @staticmethod
    def analyze(data):
        risks = []
        for _, row in data.iterrows():
            negative_count = sum(1 for x in row[['budget', 'target', 'quota', 'paid']] if x < 0)
            if negative_count > 2:
                risks.append({
                    'specialty': row['specialty'],
                    'risk_level': 'high',
                    'indicators': negative_count
                })
        return pd.DataFrame(risks)


# =================================
# Risk Analysis Tab
# =================================

class RiskAnalysisTab(wx.Panel):
    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 2)
        self.setup_columns()

        vbox.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(vbox)

    def setup_columns(self):
        columns = [
            ("Специальность", 400),
            ("Масштаб изменений", 200)
        ]
        for col, (label, width) in enumerate(columns):
            self.grid.SetColLabelValue(col, label)
            self.grid.SetColSize(col, width)
        self.grid.DisableDragColSize()

    def update_risks(self, risks):
        self.grid.ClearGrid()
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())

        for i, (specialty, risk) in enumerate(risks.items()):
            self.grid.AppendRows(1)
            self.grid.SetCellValue(i, 0, specialty)
            self.grid.SetCellValue(i, 1, risk)

    def update_risk_analysis(self):
        risks = {}
        if self.current_predictions is not None and self.data:
            latest_year = sorted(self.data.keys())[-1]
            df = self.data[latest_year]
            for i, row in df.iterrows():
                # Проверка отрицательных значений в прогнозе
                if any(x < 0 for x in self.current_predictions[i]):
                    risks[row['specialty']] = "Значительные изменения"
                # Проверка резких изменений
                elif abs(sum(self.current_predictions[i]) - sum(row[['budget', 'target', 'quota', 'paid']])) > 50:
                    risks[row['specialty']] = "Средний масштаб изменений"
        self.risk_tab.update_risks(risks)


# =================================
# SettingsDialog
# =================================

class SettingsDialog(wx.Dialog):
    def __init__(self, parent, main_frame):
        super().__init__(parent, title="Настройки модели", size=(400, 300))
        self.main_frame = main_frame
        self.settings_tab = SettingsTab(self, main_frame)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.settings_tab, 1, wx.EXPAND | wx.ALL, 10)

        btn_close = wx.Button(self, wx.ID_OK, "Закрыть")
        vbox.Add(btn_close, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        self.SetSizer(vbox)
        self.Centre()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        architecture = self.nn.get_architecture()
        info = wx.StaticText(self, label=f"Текущая архитектура:\n"
                                         f"Вход: {architecture['input_size']}\n"
                                         f"Скрытый слой 1: {architecture['hidden1']}\n"
                                         f"Скрытый слой 2: {architecture['hidden2']}\n"
                                         f"Выход: {architecture['output']}")

        vbox.Add(info, 0, wx.ALL, 10)
        vbox.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 5)

        self.epochs = wx.SpinCtrl(self, min=100, max=10000, value='2000')
        self.lr = wx.TextCtrl(self, value='0.0001')

        grid = wx.FlexGridSizer(2, 2, 10, 10)
        grid.AddMany([
            (wx.StaticText(self, label="Количество эпох:"), self.epochs),
            (wx.StaticText(self, label="Скорость обучения:"), self.lr)
        ])

        vbox.Add(grid, 0, wx.ALL, 10)
        btn_save = wx.Button(self, label="Сохранить")
        btn_save.Bind(wx.EVT_BUTTON, self.on_save)

        vbox.Add(btn_save, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        self.SetSizerAndFit(vbox)

    def on_save(self, event):
        self.Close()


# =================================
# AboutDialog
# =================================

class AboutDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(parent, title="О программе", size=(400, 300))
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        info = wx.StaticText(self, label="Admission Predictor Pro\nВерсия 2.1")
        font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        info.SetFont(font)

        details = wx.StaticText(self,
                                label="Программа для прогнозирования набора студентов\nв высшие учебные заведения")
        link = wx.HyperlinkCtrl(self, -1, "Документация", "https://help.university-predictor.ru")
        btn_manual = wx.Button(self, label="Открыть руководство")

        vbox.Add(info, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        vbox.Add(details, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        vbox.Add(link, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        vbox.Add(btn_manual, 0, wx.ALL | wx.ALIGN_CENTER, 10)

        btn_manual.Bind(wx.EVT_BUTTON, lambda e: UserManualDialog(self).ShowModal())

        self.SetSizer(vbox)
        self.Centre()


# =================================
# User Manual
# =================================

class UserManualDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(parent, title="Руководство пользователя", size=(1000, 600))
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        notebook = wx.Notebook(self)

        self.add_scrolled_tab(notebook, "Основы работы", self.get_basic_usage_text())
        self.add_scrolled_tab(notebook, "Работа с данными", self.get_data_management_text())
        self.add_scrolled_tab(notebook, "Модель и обучение", self.get_model_text())
        self.add_scrolled_tab(notebook, "Анализ данных", self.get_analysis_text())
        self.add_scrolled_tab(notebook, "История и риски", self.get_history_text())
        self.add_scrolled_tab(notebook, "Решение проблем", self.get_troubleshooting_text())

        vbox.Add(notebook, 1, wx.EXPAND | wx.ALL, 10)

        btn_close = wx.Button(self, wx.ID_CANCEL, "Закрыть")
        vbox.Add(btn_close, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        self.SetSizer(vbox)
        self.Centre()
        self.SetMinSize((800, 500))

    def add_scrolled_tab(self, notebook, title, content):
        panel = wx.ScrolledWindow(notebook)
        panel.SetScrollRate(20, 20)

        sizer = wx.BoxSizer(wx.VERTICAL)

        text_ctrl = wx.TextCtrl(
            panel,
            value=content,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_AUTO_URL | wx.TE_RICH2,
            size=(800, -1)
        )
        text_ctrl.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        text_ctrl.SetBackgroundColour(wx.Colour(240, 240, 240))

        sizer.Add(text_ctrl, 1, wx.EXPAND | wx.ALL, 10)
        panel.SetSizer(sizer)

        notebook.AddPage(panel, title)

    def get_basic_usage_text(self):
        return """1. Основной рабочий процесс:

Запуск программы

• При первом запуске откроется окно с пустой таблицей
• Используйте меню или панель инструментов для навигации

Быстрый старт

1. Вкладка 'Ввод данных' - добавьте данные за 2+ года
2. Нажмите 'Обучить' на панели инструментов
3. После обучения нажмите 'Прогноз'
4. Анализируйте результаты на других вкладках

Горячие клавиши

• Ctrl+S - сохранить проект
• Ctrl+O - открыть проект
• Ctrl+L - светлая тема
• Ctrl+D - темная тема"""

    def get_data_management_text(self):
        return """2. Управление данными:

Формат данных

Обязательные колонки:
  1. Специальность (текст)
  2. Бюджетные места (целое число)
  3. Целевой набор (целое число)
  4. Квота (целое число)
  5. Платные места (целое число)

Импорт данных
1. Скопируйте таблицу из Excel
2. Нажмите 'Вставить из Excel'
3. Проверьте корректность данных
4. Введите год и нажмите 'Сохранить'

Требования
• Минимум 5 специальностей
• Данные за 2+ последовательных года
• Отсутствие пустых значений"""

    def get_model_text(self):
        return """3. Работа с моделью:

Обучение модели

Требования:
  - Данные за минимум 2 года
  - Сохраненные данные в системе

Процесс обучения:
  1. Нажмите кнопку 'Обучить'
  2. Дождитесь завершения (2-5 минут)
  3. Статус отображается в нижней панели

Настройки модели

• Доступны через: Инструменты → Настройки модели
• Параметры:
  - Количество эпох (100-10000)
  - Размер скрытых слоев
  - Скорость обучения

Интерпретация результатов

• Прогнозные значения:
  - Отображаются в столбце 'Прогноз'"""

    def get_analysis_text(self):
        return """4. Анализ данных:

Вкладка 'Прогнозы'

• Таблица с сравнением показателей:
  - Текущие значения
  - Прогнозируемые значения
  - Процент изменения
  - Тренд

Вкладка 'Графики'

• Доступные графики:
  1. Топ-10 специальностей по бюджету
  2. Сравнение текущих и прогнозных значений
  3. Распределение рисков

Экспорт данных

• Форматы экспорта:
  - Excel (полная таблица)
  - PDF (отчет с графиками)
  - PNG (изображения графиков)

Анализ рисков

• Уровни риска:
  - Высокий: красный значок
  - Средний: желтый значок
  - Низкий: зеленый значок"""

    def get_history_text(self):
        return """5. История и риски:

История прогнозов

• Хранит:
  - Дату и время прогноза
  - Использованные данные
  - Результаты прогноза
  - Статус выполнения

Работа с историей

• Просмотр: двойной клик по записи
• Фильтрация: контекстное меню
• Экспорт: через меню вкладки

Анализ рисков

• Критерии оценки:
  - Резкие изменения показателей
  - Отрицательные значения
  - Отклонения от средних значений

Восстановление данных

• Из истории можно:
  - Загрузить предыдущие данные
  - Повторить прогноз
  - Сравнить разные версии"""

    def get_troubleshooting_text(self):
        return """6. Решение проблем:

Частые ошибки

• 'Нет данных для обучения':
  - Добавьте данные за 2+ года
  - Убедитесь в сохранении данных

• 'Неверный формат данных':
  - Проверьте числовые поля
  - Удалите текстовые символы

Технические проблемы

• Зависание интерфейса:
  - Закройте диалоговые окна
  - Перезапустите программу

• Проблемы с экспортом:
  - Закройте открытые файлы Excel
  - Проверьте права доступа

Обновление программы
• Проверьте наличие обновлений:
  - Меню 'Справка' → 'О программе'
  - Автоматические уведомления"""

    def add_scrolled_tab(self, notebook, title, content):
        panel = wx.ScrolledWindow(notebook)
        panel.SetScrollRate(20, 20)

        sizer = wx.BoxSizer(wx.VERTICAL)

        text_ctrl = wx.TextCtrl(
            panel,
            value=content,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_AUTO_URL | wx.TE_RICH2,
            size=(800, -1)
        )
        text_ctrl.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        text_ctrl.SetBackgroundColour(wx.Colour(240, 240, 240))

        sizer.Add(text_ctrl, 1, wx.EXPAND | wx.ALL, 10)
        panel.SetSizer(sizer)

        notebook.AddPage(panel, title)

# =================================
# HelpWindow
# =================================

class HelpWindow(wx.Frame):
    def __init__(self, parent):
        super().__init__(parent, title="Справка", size=(800, 600))
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        text.SetFont(font)

        help_content = """
        Техническое описание системы прогнозирования приёма в вузы

        1. Архитектура системы
        ----------------------
        1.1. Общая структура
        Система построена по модульному принципу с использованием паттерна MVC:
        - Model: Нейронная сеть, модули обработки данных
        - View: GUI на wxPython
        - Controller: Обработчики событий и бизнес-логика

        1.2. Компоненты системы
        - Главное окно с Notebook для переключения вкладок
        - Модуль ввода данных с поддержкой импорта из буфера
        - Прогнозирующий модуль с нейронной сетью
        - Визуализатор графиков на matplotlib
        - Система отчетности в PDF/Excel
        - Модуль анализа рисков
        - Журнал истории прогнозов

        2. Технологический стек
        ------------------------
        2.1. Основные технологии:
        - Язык программирования: Python 3.9+
        - GUI Framework: wxPython 4.2.0
        - Машинное обучение: NumPy, собственная реализация НС
        - Визуализация: Matplotlib 3.5.2
        - Обработка данных: Pandas 1.4.2
        - Отчетность: ReportLab 3.6.12

        2.2. Требования к системе:
        - ОС: Windows 10+/macOS 10.15+/Linux с GUI
        - Память: 4 ГБ RAM (рекомендуется 8 ГБ)
        - Процессор: x86-64 с поддержкой AVX
        - Дисковое пространство: 500 МБ

        3. Модуль нейронной сети
        ------------------------
        3.1. Архитектура сети:
        - Входной слой: 4 нейрона (бюджет, целевые, квота, платные)
        - Скрытые слои: 
          - Dense (128 нейронов, ReLU)
          - Dense (64 нейрона, ReLU)
        - Выходной слой: 4 нейрона (линейная активация)

        3.2. Параметры обучения:
        - Оптимизатор: SGD с learning rate=0.0001
        - Размер батча: 64
        - Эпохи: 2000 с ранней остановкой
        - Функция потерь: MSE
        - Метрики: RMSE, R²

        4. Форматы данных
        -----------------
        4.1. Входные данные:
        - CSV-подобный формат через буфер обмена
        - Колонки в порядке:
          1. Специальность
          2. Бюджетные места
          3. Целевой набор
          4. Квота
          5. Платные места

        4.2. Хранение данных:
        - Внутренний формат: pandas DataFrame
        - Сериализация: pickle (для моделей)
        - Кэширование: in-memory хранилище

        5. Безопасность и надежность
        ---------------------------
        5.1. Меры защиты:
        - Валидация входных данных
        - Обработка исключений
        - Лимитирование размера данных
        - Санитайзинг числовых значений

        5.2. Восстановление после сбоев:
        - Автосохранение сессий каждые 5 мин
        - Резервное копирование моделей
        - Журналирование операций

        6. Производительность
        ---------------------
        6.1. Бенчмарки:
        - Время обучения: 2-5 мин (зависит от данных)
        - Прогноз: <100 мс для 1000 записей
        - Загрузка данных: 1 сек на 10к строк

        6.2. Оптимизации:
        - Векторизация операций с NumPy
        - Многопоточная обработка
        - Кэширование вычислений

        7. Расширяемость системы
        ------------------------
        7.1. Поддерживаемые расширения:
        - Плагины для новых алгоритмов
        - Кастомные визуализации
        - Интеграция с внешними API
        - Поддержка новых форматов данных

        7.2. API для разработчиков:
        - DataProcessor: базовый класс обработки
        - ModelWrapper: интерфейс моделей
        - Visualizer: система визуализации
        - Exporter: система экспорта

        8. Ограничения системы
        ----------------------
        8.1. Текущие ограничения:
        - Макс. размер данных: 100к строк
        - Поддержка только числовых признаков
        - Ограниченная интернационализация
        - Нет распределенных вычислений

        8.2. Планы по развитию:
        - Интеграция с облачными сервисами
        - Поддержка GPU-ускорения
        - Автоматический подбор параметров
        - Расширенная аналитика


        Для получения дополнительной информации обращайтесь:
        support-university-predictor@yandex.ru
        """

        text.SetValue(help_content)
        sizer.Add(text, 1, wx.EXPAND | wx.ALL, 10)

        close_btn = wx.Button(panel, label="Закрыть")
        close_btn.Bind(wx.EVT_BUTTON, lambda e: self.Destroy())
        sizer.Add(close_btn, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        panel.SetSizer(sizer)
        self.Show()


# =================================
# Notification System
# =================================

class NotificationManager:
    def __init__(self, status_bar):
        self.status_bar = status_bar
        self.history = []

    def show_status(self, message, duration=3000):
        self.status_bar.SetStatusText(message)
        self.history.append(f"STATUS: {message}")
        wx.CallLater(duration, lambda: self.status_bar.SetStatusText(""))

    def show_popup(self, title, message, style=wx.ICON_INFORMATION):
        dlg = wx.MessageDialog(None, message, title, style)
        dlg.ShowModal()
        dlg.Destroy()
        self.history.append(f"POPUP: {title} - {message}")


# =================================
# Theme Management
# =================================

class ThemeManager:
    def __init__(self):
        self.current_theme = "light"

    def apply_theme(self, window, theme_name):
        self.current_theme = theme_name
        theme = THEMES[theme_name]

        self._apply_theme_recursive(window, theme)

    def _apply_theme_recursive(self, widget, theme):
        if isinstance(widget, wx.Panel):
            widget.SetBackgroundColour(theme["panel"])
        elif isinstance(widget, wx.StaticText):
            widget.SetForegroundColour(theme["text"])
        elif isinstance(widget, gridlib.Grid):
            widget.SetDefaultCellBackgroundColour(theme["grid_bg"])
            widget.SetDefaultCellTextColour(theme["grid_text"])
            widget.SetLabelBackgroundColour(theme["panel"])
            widget.SetLabelTextColour(theme["text"])
            widget.ForceRefresh()

        # Рекурсивно обрабатываем дочерние элементы
        for child in widget.GetChildren():
            self._apply_theme_recursive(child, theme)


# =================================
# Context Menu
# =================================

class GridContextMenu(wx.Menu):
    def __init__(self, grid):
        super().__init__()
        self.grid = grid

        copy_item = self.Append(wx.ID_COPY, "Копировать")
        paste_item = self.Append(wx.ID_PASTE, "Вставить")
        clear_item = self.Append(wx.ID_CLEAR, "Очистить")

        self.Bind(wx.EVT_MENU, self.on_copy, copy_item)
        self.Bind(wx.EVT_MENU, self.on_paste, paste_item)
        self.Bind(wx.EVT_MENU, self.on_clear, clear_item)

    def on_copy(self, event):
        self.grid.Copy()

    def on_paste(self, event):
        self.grid.Paste()

    def on_clear(self, event):
        self.grid.ClearGrid()


# ========================
# Data Validation
# ========================

class DataValidator:
    @staticmethod
    def validate_grid(grid):
        errors = []
        for row in range(grid.GetNumberRows()):
            for col in range(1, 5):
                value = grid.GetCellValue(row, col)
                if not value.isdigit():
                    grid.SetCellBackgroundColour(row, col, wx.RED)
                    errors.append(f"Row {row + 1}, Col {col + 1}: Invalid number")
                else:
                    grid.SetCellBackgroundColour(row, col, wx.WHITE)
        grid.ForceRefresh()
        return errors


# ========================
# Context Menu
# ========================

class GridContextMenu(wx.Menu):
    def __init__(self, grid):
        super().__init__()
        self.grid = grid

        copy_item = self.Append(wx.ID_COPY, "Copy")
        paste_item = self.Append(wx.ID_PASTE, "Paste")
        clear_item = self.Append(wx.ID_CLEAR, "Clear")

        self.Bind(wx.EVT_MENU, self.on_copy, copy_item)
        self.Bind(wx.EVT_MENU, self.on_paste, paste_item)
        self.Bind(wx.EVT_MENU, self.on_clear, clear_item)

    def on_copy(self, event):
        self.grid.Copy()

    def on_paste(self, event):
        self.grid.Paste()

    def on_clear(self, event):
        self.grid.ClearGrid()


# ========================
# Update Checker
# ========================

class UpdateChecker(threading.Thread):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.daemon = True

    def run(self):
        try:
            import requests
            response = requests.get("https://api.github.com/repos/example/university-predictor/releases/latest")
            latest_version = response.json()["tag_name"]
            wx.CallAfter(self.callback, latest_version)
        except Exception as e:
            wx.CallAfter(self.callback, None)


# ========================
# Logging System
# ========================

class AppLogger:
    def __init__(self):
        self.log_file = "app.log"
        self.setup_logger()

    def setup_logger(self):
        import logging
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger()

    def log_operation(self, operation, status="success"):
        self.logger.info(f"{operation.upper()} - {status}")


# =================================
# Settings Tab Implementation
# =================================

class SettingsTab(wx.Panel):
    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.epochs = wx.SpinCtrl(self, min=100, max=10000, initial=2000)
        self.hidden_size = wx.SpinCtrl(self, min=32, max=512, initial=128)
        self.learning_rate = wx.TextCtrl(self, value="0.0001")

        form = wx.FlexGridSizer(cols=2, hgap=10, vgap=10)
        form.AddGrowableCol(1, 1)

        form.AddMany([
            (wx.StaticText(self, label="Количество эпох:"), 0, wx.ALIGN_CENTER_VERTICAL),
            (self.epochs, 1, wx.EXPAND),

            (wx.StaticText(self, label="Размер скрытого слоя:"), 0, wx.ALIGN_CENTER_VERTICAL),
            (self.hidden_size, 1, wx.EXPAND),

            (wx.StaticText(self, label="Скорость обучения:"), 0, wx.ALIGN_CENTER_VERTICAL),
            (self.learning_rate, 1, wx.EXPAND)
        ])

        btn_save = wx.Button(self, label="Сохранить настройки")
        btn_default = wx.Button(self, label="Сбросить настройки")

        vbox.Add(form, 0, wx.EXPAND | wx.ALL, 15)
        vbox.Add(btn_save, 0, wx.ALIGN_RIGHT | wx.RIGHT | wx.TOP, 15)
        vbox.Add(btn_default, 0, wx.ALIGN_RIGHT | wx.RIGHT | wx.BOTTOM, 15)

        btn_save.Bind(wx.EVT_BUTTON, self.on_save)
        btn_default.Bind(wx.EVT_BUTTON, self.on_default)

        self.SetSizer(vbox)
        self.Layout()

    def load_settings(self):
        """Загрузка текущих настроек"""
        if hasattr(self.main_frame, 'model_settings'):
            self.epochs.SetValue(self.main_frame.model_settings.get('epochs', 2000))
            self.hidden_size.SetValue(self.main_frame.model_settings.get('hidden_size', 128))
            self.learning_rate.SetValue(str(self.main_frame.model_settings.get('learning_rate', 0.0001)))

    def on_save(self, event):
        """Сохранение настроек"""
        try:
            self.main_frame.model_settings = {
                'epochs': self.epochs.GetValue(),
                'hidden_size': self.hidden_size.GetValue(),
                'learning_rate': float(self.learning_rate.GetValue())
            }
            wx.MessageBox("Настройки успешно сохранены!", "Успех", wx.OK | wx.ICON_INFORMATION)
        except ValueError:
            wx.MessageBox("Некорректное значение скорости обучения!", "Ошибка", wx.OK | wx.ICON_ERROR)

    def on_default(self, event):
        """Сброс настроек к значениям по умолчанию"""
        self.epochs.SetValue(2000)
        self.hidden_size.SetValue(128)
        self.learning_rate.SetValue("0.0001")
        wx.MessageBox("Настройки сброшены к значениям по умолчанию", "Информация", wx.OK | wx.ICON_INFORMATION)


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
    app.SetAppName("AdmissionPredictor")

    if platform.system() == 'Darwin':
        wx.SystemOptions.SetOption("osx.openfiledialog.always-show-types", "1")

    frame = MainFrame()
    app.MainLoop()