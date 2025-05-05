import wx
import wx.grid as gridlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('WXAgg')
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg
from sklearn.metrics import r2_score
import time
import threading
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)
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
        self.current_mode = 'main'  # 'main' или 'city'
        self.init_ui()
        self.SetBackgroundColour(wx.WHITE)

    def init_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        mode_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_main = wx.Button(self, label="Основные данные")
        self.btn_city = wx.Button(self, label="Городские данные")
        mode_sizer.Add(self.btn_main, 0, wx.RIGHT, 10)
        mode_sizer.Add(self.btn_city, 0)

        # Грид
        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 6)
        self.setup_columns()

        # Отдельная панель для кнопок
        button_panel = wx.Panel(self)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_excel = wx.Button(button_panel, label="Excel")
        self.btn_help = wx.Button(button_panel, label="Справка")
        button_sizer.Add(self.btn_excel, 0, wx.RIGHT, 10)
        button_sizer.Add(self.btn_help, 0, wx.LEFT, 10)
        button_panel.SetSizer(button_sizer)

        main_sizer.Add(mode_sizer, 0, wx.ALL, 5)
        main_sizer.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(button_panel, 0, wx.ALIGN_RIGHT | wx.ALL, 5)
        self.SetSizerAndFit(main_sizer)

        self.btn_main.Bind(wx.EVT_BUTTON, lambda e: self.switch_mode('main'))
        self.btn_city.Bind(wx.EVT_BUTTON, lambda e: self.switch_mode('city'))
        self.btn_excel.Bind(wx.EVT_BUTTON, self.export_excel)
        self.btn_help.Bind(wx.EVT_BUTTON, self.show_help)

    def switch_mode(self, mode):
        self.current_mode = mode
        if mode == 'main':
            self.btn_main.SetBackgroundColour(wx.Colour(200, 200, 255))
            self.btn_city.SetBackgroundColour(wx.NullColour)
            if hasattr(self.main_frame, 'current_predictions') and hasattr(self.main_frame,
                                                                           'data') and self.main_frame.data:
                try:
                    latest_year = sorted(self.main_frame.data.keys())[-1]
                    self.update_predictions(
                        self.main_frame.data[latest_year],
                        self.main_frame.current_predictions
                    )
                except IndexError:
                    wx.MessageBox("Нет основных данных для отображения", "Информация", wx.OK | wx.ICON_INFORMATION)
            else:
                self.grid.ClearGrid()
                wx.MessageBox("Нет основных данных для отображения", "Информация", wx.OK | wx.ICON_INFORMATION)
        else:
            self.btn_city.SetBackgroundColour(wx.Colour(200, 200, 255))
            self.btn_main.SetBackgroundColour(wx.NullColour)
            if hasattr(self.main_frame, 'current_city_predictions') and hasattr(self.main_frame,
                                                                                'city_data') and self.main_frame.city_data:
                try:
                    latest_year = sorted(self.main_frame.city_data.keys())[-1]
                    self.update_city_predictions(
                        self.main_frame.city_data[latest_year],
                        self.main_frame.current_city_predictions
                    )
                except IndexError:
                    wx.MessageBox("Нет городских данных для отображения", "Информация", wx.OK | wx.ICON_INFORMATION)
            else:
                self.grid.ClearGrid()
                wx.MessageBox("Нет городских данных для отображения", "Информация", wx.OK | wx.ICON_INFORMATION)
        self.Refresh()

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
            # Очищаем предыдущие данные
            current_row_count = self.grid.GetNumberRows()
            if current_row_count > 0:
                self.grid.DeleteRows(0, current_row_count)

            # Создаем новые строки
            required_rows = len(data)
            if required_rows > 0:
                self.grid.AppendRows(required_rows)

            # Рассчитываем показатели
            data['total_current'] = data[['budget', 'target', 'quota', 'paid']].sum(axis=1)
            data['total_predicted'] = [sum(pred) for pred in predictions]
            data['change_abs'] = data['total_predicted'] - data['total_current']
            data['change_pct'] = (data['change_abs'] / data['total_current'] * 100).fillna(0)
            data['trend'] = np.where(data['change_abs'] >= 0, '▲ Рост', '▼ Снижение')

            # Заполняем таблицу
            for idx, row in data.iterrows():
                if idx >= self.grid.GetNumberRows():
                    break

                self.grid.SetCellValue(idx, 0, str(row['specialty']))
                self.grid.SetCellValue(idx, 1, f"{row['total_current']:.0f}")
                self.grid.SetCellValue(idx, 2, f"{row['total_predicted']:.0f}")
                self.grid.SetCellValue(idx, 3, f"{row['change_pct']:.1f}%")
                self.grid.SetCellValue(idx, 4, f"{row['change_abs']:.0f}")
                self.grid.SetCellValue(idx, 5, row['trend'])

                # Подсветка ячеек
                for col in [3, 4, 5]:
                    if row['change_abs'] > 0:
                        self.grid.SetCellBackgroundColour(idx, col, wx.Colour(220, 255, 220))
                    elif row['change_abs'] < 0:
                        self.grid.SetCellBackgroundColour(idx, col, wx.Colour(255, 220, 220))

            self.grid.AutoSizeColumns()
            self.grid.ForceRefresh()

        except Exception as e:
            error_msg = f"Ошибка обновления прогнозов: {str(e)}"
            print(error_msg)
            wx.MessageBox(error_msg, "Ошибка", wx.OK | wx.ICON_ERROR)
            self.grid.ClearGrid()

    def update_city_predictions(self, data, predictions):
        try:
            current_row_count = self.grid.GetNumberRows()
            required_rows = len(data)

            if current_row_count > required_rows:
                self.grid.DeleteRows(required_rows, current_row_count - required_rows)
            elif current_row_count < required_rows:
                self.grid.AppendRows(required_rows - current_row_count)

            columns = [
                ("Город", 300),
                ("Текущее очное", 150),
                ("Прогноз очное", 150),
                ("Текущее заочное", 150),
                ("Прогноз заочное", 150),
                ("Изменение", 100)
            ]
            for col, (label, width) in enumerate(columns):
                self.grid.SetColLabelValue(col, label)
                self.grid.SetColSize(col, width)

            for i, row in data.iterrows():
                if i >= self.grid.GetNumberRows():
                    break

                current_full = row.get('full_time', 0)
                current_part = row.get('part_time', 0)
                pred_full = predictions[i][0] if i < len(predictions) else 0
                pred_part = predictions[i][1] if i < len(predictions) else 0

                total_current = current_full + current_part
                total_pred = pred_full + pred_part

                change = total_pred - total_current
                change_percent = (change / total_current * 100) if total_current != 0 else 0

                cols = [
                    str(row.get('city', '')),
                    f"{current_full:.0f}",
                    f"{pred_full:.0f}",
                    f"{current_part:.0f}",
                    f"{pred_part:.0f}",
                    f"{change:.0f} ({change_percent:.1f}%)"
                ]

                for col_idx in range(6):
                    if col_idx < len(cols):
                        self.grid.SetCellValue(i, col_idx, cols[col_idx])

            self.grid.AutoSizeColumns()
            self.grid.ForceRefresh()

        except Exception as e:
            print(f"Ошибка обновления городских прогнозов: {str(e)}")
            wx.MessageBox(f"Ошибка обновления городских данных: {str(e)}", "Ошибка", wx.OK | wx.ICON_ERROR)
            self.grid.ClearGrid()

    def calculate_percent_change(self, row, predictions, index):
        try:
            current = sum([row.get('budget', 0), row.get('target', 0), row.get('quota', 0), row.get('paid', 0)])
            predicted = sum(predictions[index]) if index < len(predictions) else 0

            if current == 0:
                return "N/A"

            change = ((predicted - current) / current * 100)
            return f"{change:.0f}%"

        except Exception as e:
            print(f"Ошибка расчета процента: {str(e)}")
            return "Error"

    def show_history(self, event):
        if self.main_frame.history.get_last():
            data = self.main_frame.history.get_last()['data']
            self.update_predictions(data, data)  # Упрощенный пример
        else:
            wx.MessageBox("История прогнозов пуста!", "Информация", wx.OK | wx.ICON_INFORMATION)

    def export_excel(self, event):
        df = self.get_grid_data()
        mode = "основных" if self.current_mode == 'main' else "городских"
        with wx.FileDialog(
                self,
                f"Сохранить {mode} данные в Excel",
                wildcard="Excel files (*.xlsx)|*.xlsx",
                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return
            df.to_excel(fd.GetPath(), index=False)
            wx.MessageBox(f"{mode.capitalize()} данные экспортированы в Excel", "Успех", wx.OK | wx.ICON_INFORMATION)

    def export_pdf(self, event):
        pass

    def export_pdf_report(self, df, filename):
        pass

    def get_grid_data(self):
        data = []
        if self.current_mode == 'main':
            columns = ['Специальность', 'Текущий набор', 'Прогноз', 'Изменение (%)', 'Изменение', 'Тренд']
        else:
            columns = ['Город', 'Текущее очное', 'Прогноз очное', 'Текущее заочное', 'Прогноз заочное', 'Изменение']

        for row in range(self.grid.GetNumberRows()):
            row_data = []
            for col in range(6):
                row_data.append(self.grid.GetCellValue(row, col))
            data.append(row_data)

        return pd.DataFrame(data, columns=columns)

    def export_pdf_report(self, df, filename):
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []

        elements.append(Paragraph("Отчет по прогнозам приема", getSampleStyleSheet()['Title']))
        elements.append(Spacer(1, 20))

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
            self.main_frame.graph_tab.figure.savefig(graph_path, bbox_inches='tight')
            img = Image(graph_path, width=400, height=300)
            elements.append(img)
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
        self.current_graph = 0
        self.init_ui()

    def init_ui(self):
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar2WxAgg(self.canvas)

        self.btn_prev = wx.Button(self, label="← Пред")
        self.btn_next = wx.Button(self, label="След →")

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(self.btn_prev, 0, wx.RIGHT, 10)
        btn_sizer.Add(self.btn_next, 0)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.toolbar, 0, wx.EXPAND)
        main_sizer.Add(self.canvas, 1, wx.EXPAND)
        main_sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        self.SetSizer(main_sizer)

        self.btn_prev.Bind(wx.EVT_BUTTON, self.on_prev)
        self.btn_next.Bind(wx.EVT_BUTTON, self.on_next)
        self.update_graphs()

    def update_graphs(self):
        self.figure.clear()

        if self.current_graph == 0:
            self.plot_main()
        elif self.current_graph == 1:
            self.plot_distribution()
        elif self.current_graph == 2:
            self.plot_comparison()
        elif self.current_graph == 3:
            self.plot_trend()

            self.figure.tight_layout(rect=[0.1, 0.1, 0.9, 0.85])  # Оставляем место для легенды
            self.canvas.draw()

        self.canvas.draw()

    def plot_main(self):
        ax = self.figure.add_subplot(111)
        if not self.main_frame.data:
            return

        latest_year = sorted(self.main_frame.data.keys())[-1]
        data = self.main_frame.data[latest_year]

        data['total'] = data[['budget', 'target', 'quota', 'paid']].sum(axis=1)

        top_data = data.nlargest(10, 'total')

        specialties = top_data['specialty'].tolist()
        current = top_data['total'].values
        predicted = [sum(pred) for pred in self.main_frame.current_predictions[:10]]

        x = np.arange(len(specialties))
        ax.bar(x - 0.2, current, 0.4, label='Текущий', color='#1f77b4')
        ax.bar(x + 0.2, predicted, 0.4, label='Прогноз', color='#ff7f0e')

        ax.set_xticks(x)
        ax.set_xticklabels(specialties, rotation=45, ha='right')
        ax.legend()
        ax.set_title('Топ 10 специальностей по общему количеству мест')
        ax.grid(True, linestyle='--', alpha=0.7)

        for i, (curr, pred) in enumerate(zip(current, predicted)):
            ax.text(i - 0.3, curr + 5, f'{curr:.0f}', ha='center')
            ax.text(i + 0.3, pred + 5, f'{pred:.0f}', ha='center')

    def plot_distribution(self):
        ax = self.figure.add_subplot(111)
        if self.main_frame.current_predictions is None:
            return

        changes = []

        for i, row in self.main_frame.data[sorted(self.main_frame.data.keys())[-1]].iloc[1:].iterrows():
            current = sum([row['budget'], row['target'], row['quota'], row['paid']])
            predicted = sum(self.main_frame.current_predictions[i])
            changes.append(predicted - current)

        ax.hist(changes, bins=15, color='skyblue', edgecolor='black')
        ax.set_title('Распределение изменений')
        ax.set_xlabel('Изменение количества мест')
        ax.set_ylabel('Количество специальностей')
        ax.grid(True)

    def plot_comparison(self):
        ax = self.figure.add_subplot(111)
        if self.main_frame.current_predictions is None:
            return

        data = self.main_frame.data[sorted(self.main_frame.data.keys())[-1]].iloc[1:]
        total_budget = data['budget'].sum()
        total_paid = data['paid'].sum()

        labels = ['Бюджетные', 'Платные']
        sizes = [total_budget, total_paid]

        ax.pie(sizes, labels=labels, autopct='%1.1f%%',
               startangle=90, colors=['#66b3ff', '#99ff99'])
        ax.axis('equal')
        ax.set_title('Соотношение бюджетных и платных мест')

    def plot_trend(self):
        ax = self.figure.add_subplot(111)

        if len(self.main_frame.data) < 2:
            ax.set_title("Недостаточно данных для построения тренда")
            return

        try:
            years = sorted(self.main_frame.data.keys())
            trends = []

            for year in years:
                total = self.main_frame.data[year][['budget', 'target', 'quota', 'paid']].sum().sum()
                trends.append(total)

            str_years = [str(year) for year in years]

            ax.plot(str_years, trends, marker='o', linestyle='-', color='green')
            ax.set_title('Динамика общего набора по годам')
            ax.grid(True)
            ax.set_xlabel('Год')
            ax.set_ylabel('Общее количество мест')

        except Exception as e:
            print(f"Ошибка построения тренда: {str(e)}")
            ax.set_title("Ошибка построения тренда")

    def save_plot(self, filename, title):
        self.figure.savefig(filename, bbox_inches='tight')
        return filename

    def on_prev(self, event):
        self.current_graph = (self.current_graph - 1) % 3
        self.update_graphs()

    def on_next(self, event):
        self.current_graph = (self.current_graph + 1) % 3
        self.update_graphs()


# =================================
# Neural Network Implementation
# =================================

class NeuralNetwork:
    def __init__(self, input_size, hidden_size=128, output_size=4):
        np.random.seed(42)

        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
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

        notebook.AddPage(self.input_tab, "Ввод данных")
        notebook.AddPage(self.prediction_tab, "Прогнозы")
        notebook.AddPage(self.graph_tab, "Графики")
        notebook.AddPage(self.history_tab, "История")

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
        pass

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
            self.history_tab.grid
        ]

        for grid in grids:
            grid.Bind(wx.EVT_CONTEXT_MENU, self.show_context_menu)

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

            menu = GridContextMenu(grid)

            grid.PopupMenu(menu)

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

        has_main_data = hasattr(self, 'data') and len(self.data) >= 2
        has_city_data = hasattr(self, 'city_data') and len(self.city_data) >= 2

        if not has_main_data and not has_city_data:
            self.notifier.show_popup("Ошибка",
                                     "Для обучения нужно:\n"
                                     "- Основные данные за 2+ года\n"
                                     "- Или городские данные за 2+ года",
                                     wx.ICON_ERROR)
            return

        self.loading_dialog = LoadingScreen(self)
        self.notifier.show_status("Начато обучение модели...")

        def train_thread():
            try:

                if has_main_data:
                    X_main, y_main = self.prepare_main_training_data()
                    print(f"[MAIN DATA] Training shape - X: {X_main.shape}, y: {y_main.shape}")

                    self.nn_main = NeuralNetwork(
                        input_size=X_main.shape[1],
                        hidden_size=self.model_settings['hidden_size'],
                        output_size=y_main.shape[1]
                    )
                    self.nn_main.train(X_main, y_main,
                                       epochs=self.model_settings['epochs'],
                                       lr=self.model_settings['learning_rate'])

                if has_city_data:
                    X_city, y_city = self.prepare_city_training_data()
                    print(f"[CITY DATA] Training shape - X: {X_city.shape}, y: {y_city.shape}")

                    self.nn_city = NeuralNetwork(
                        input_size=X_city.shape[1],
                        hidden_size=self.model_settings['hidden_size'],
                        output_size=y_city.shape[1]
                    )
                    self.nn_city.train(X_city, y_city,
                                       epochs=self.model_settings['epochs'],
                                       lr=self.model_settings['learning_rate'])

                wx.CallAfter(self._training_finished)

            except Exception as e:
                error_msg = f"Ошибка обучения: {str(e)}"
                print(f"[ERROR] {error_msg}")
                wx.CallAfter(self._training_failed, error_msg)
            finally:
                wx.CallAfter(self._cleanup_loading_dialog)

        threading.Thread(target=train_thread, daemon=True).start()

    def prepare_main_training_data(self):
        """Подготовка данных по специальностям (4 колонки)"""
        if len(self.data) < 2:
            raise ValueError("Нужны основные данные за 2+ года")

        features, targets = [], []
        years = sorted(self.data.keys())

        for i in range(len(years) - 1):
            current = self.data[years[i]]
            next_year = self.data[years[i + 1]]

            merged = pd.merge(current, next_year, on='specialty',
                              suffixes=('_current', '_next'))

            features.append(merged[['budget_current', 'target_current',
                                    'quota_current', 'paid_current']].values)
            targets.append(merged[['budget_next', 'target_next',
                                   'quota_next', 'paid_next']].values)

        return np.vstack(features), np.vstack(targets)

    def prepare_city_training_data(self):
        """Подготовка городских данных (2 колонки)"""
        if len(self.city_data) < 2:
            raise ValueError("Нужны городские данные за 2+ года")

        features, targets = [], []
        years = sorted(self.city_data.keys())

        for i in range(len(years) - 1):
            current = self.city_data[years[i]]
            next_year = self.city_data[years[i + 1]]

            merged = pd.merge(current, next_year, on='city',
                              suffixes=('_current', '_next'))

            features.append(merged[['full_time_current', 'part_time_current']].values)
            targets.append(merged[['full_time_next', 'part_time_next']].values)

        return np.vstack(features), np.vstack(targets)

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
        try:
            # Прогнозирование для основных данных
            if hasattr(self, 'nn_main') and hasattr(self, 'data'):
                latest_year = sorted(self.data.keys())[-1]
                X_main = self.data[latest_year][['budget', 'target', 'quota', 'paid']].values
                self.current_predictions = self.nn_main.predict(X_main)

            # Прогнозирование для городских данных
            if hasattr(self, 'nn_city') and hasattr(self, 'city_data'):
                latest_year = sorted(self.city_data.keys())[-1]
                X_city = self.city_data[latest_year][['full_time', 'part_time']].values
                self.current_city_predictions = self.nn_city.predict(X_city)

            wx.CallAfter(self._update_predictions_display)

        except Exception as e:
            self.notifier.show_popup("Ошибка", f"Ошибка прогнозирования: {str(e)}", wx.ICON_ERROR)

    def _update_predictions_display(self):
        if hasattr(self, 'prediction_tab'):
            if self.prediction_tab.current_mode == 'main' and hasattr(self, 'current_predictions'):
                latest_year = sorted(self.data.keys())[-1]
                current_data = self.data[latest_year]
                self.prediction_tab.update_predictions(
                    current_data,
                    self.current_predictions
                )

                self.update_history(current_data, self.current_predictions, 'specialty')

            elif self.prediction_tab.current_mode == 'city' and hasattr(self, 'current_city_predictions'):
                latest_year = sorted(self.city_data.keys())[-1]
                current_data = self.city_data[latest_year]
                self.prediction_tab.update_city_predictions(
                    current_data,
                    self.current_city_predictions
                )

                self.update_history(current_data, self.current_city_predictions, 'city')

        if hasattr(self, 'graph_tab'):
            self.graph_tab.update_graphs()

        if hasattr(self, 'graph_tab'):
            self.graph_tab.update_graphs()

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

    def update_history(self, data, predictions, data_type='specialty'):
        """Сохраняет результаты прогнозирования в историю"""
        if not hasattr(self, 'history'):
            self.history = []  # Инициализируем список истории, если его нет

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        if data_type == 'specialty':
            for i, row in data.iterrows():
                if i < len(predictions):
                    self.history.append({
                        'timestamp': timestamp,
                        'type': 'specialty',
                        'name': row['specialty'],
                        'prediction': sum(predictions[i]),
                        'status': "Успешно" if all(x >= 0 for x in predictions[i]) else "Ошибка"
                    })
        else:  # city data
            for i, row in data.iterrows():
                if i < len(predictions):
                    self.history.append({
                        'timestamp': timestamp,
                        'type': 'city',
                        'name': row['city'],
                        'prediction': f"{predictions[i][0]} (оч.), {predictions[i][1]} (заоч.)",
                        'status': "Успешно" if all(x >= 0 for x in predictions[i]) else "Ошибка"
                    })

        if len(self.history) > 100:
            self.history = self.history[-100:]

        if hasattr(self, 'history_tab'):
            wx.CallAfter(self.history_tab.update_history_display)

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
        self.notebook = wx.Notebook(self)

        self.main_tab = self.create_main_tab()
        self.city_tab = self.create_city_tab()

        self.notebook.AddPage(self.main_tab, "Основные данные")
        self.notebook.AddPage(self.city_tab, "Данные по городам")

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(vbox)

    def create_main_tab(self):
        panel = wx.Panel(self.notebook)
        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.year_entry = wx.TextCtrl(panel, size=(140, -1))
        btn_paste = wx.Button(panel, label="Вставить из Excel", size=(160, 32))
        btn_save = wx.Button(panel, label="Сохранить данные", size=(160, 32))
        btn_clear = wx.Button(panel, label="Очистить таблицу", size=(160, 32))
        btn_settings = wx.Button(panel, label="Настройки", size=(160, 32))
        btn_manual = wx.Button(panel, label="Руководство", size=(160, 32))

        hbox.Add(self.year_entry, 0, wx.RIGHT, 20)
        hbox.Add(btn_paste, 0, wx.RIGHT, 10)
        hbox.Add(btn_save, 0, wx.RIGHT, 10)
        hbox.Add(btn_clear, 0, wx.RIGHT, 10)
        hbox.Add(btn_settings, 0, wx.RIGHT, 10)
        hbox.Add(btn_manual, 0)

        self.grid = gridlib.Grid(panel)
        self.grid.CreateGrid(0, 5)
        self.setup_main_columns()

        vbox.Add(hbox, 0, wx.EXPAND | wx.ALL, 10)
        vbox.Add(self.grid, 1, wx.EXPAND)
        panel.SetSizer(vbox)

        btn_paste.Bind(wx.EVT_BUTTON, self.on_paste)
        btn_save.Bind(wx.EVT_BUTTON, self.on_save)
        btn_clear.Bind(wx.EVT_BUTTON, self.on_clear)
        btn_settings.Bind(wx.EVT_BUTTON, self.show_settings)
        btn_manual.Bind(wx.EVT_BUTTON, self.show_manual)

        return panel

    def create_city_tab(self):
        panel = wx.Panel(self.notebook)
        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.city_year_entry = wx.TextCtrl(panel, size=(140, -1))
        btn_city_paste = wx.Button(panel, label="Вставить из Excel", size=(160, 32))
        btn_city_save = wx.Button(panel, label="Сохранить данные", size=(160, 32))
        btn_city_clear = wx.Button(panel, label="Очистить таблицу", size=(160, 32))

        hbox.Add(self.city_year_entry, 0, wx.RIGHT, 20)
        hbox.Add(btn_city_paste, 0, wx.RIGHT, 10)
        hbox.Add(btn_city_save, 0, wx.RIGHT, 10)
        hbox.Add(btn_city_clear, 0)

        self.city_grid = gridlib.Grid(panel)
        self.city_grid.CreateGrid(0, 3)
        self.setup_city_columns()

        vbox.Add(hbox, 0, wx.EXPAND | wx.ALL, 10)
        vbox.Add(self.city_grid, 1, wx.EXPAND)
        panel.SetSizer(vbox)

        btn_city_paste.Bind(wx.EVT_BUTTON, self.on_city_paste)
        btn_city_save.Bind(wx.EVT_BUTTON, self.on_city_save)
        btn_city_clear.Bind(wx.EVT_BUTTON, self.on_city_clear)

        return panel

    def setup_main_columns(self):
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

    def setup_city_columns(self):
        theme = THEMES[self.GetTopLevelParent().current_theme]
        self.city_grid.SetDefaultCellBackgroundColour(theme["grid_bg"])
        self.city_grid.SetDefaultCellTextColour(theme["grid_text"])

        columns = [
            ("Город", 300),
            ("Очное отделение", 200),
            ("Заочное отделение", 200)
        ]
        for col, (label, width) in enumerate(columns):
            self.city_grid.SetColLabelValue(col, label)
            self.city_grid.SetColSize(col, width)
        self.city_grid.DisableDragColSize()

    def on_paste(self, event):
        clipboard = wx.Clipboard.Get()
        if clipboard.Open():
            data_obj = wx.TextDataObject()
            if clipboard.GetData(data_obj):
                self.load_data(data_obj.GetText())
            clipboard.Close()
            DataValidator.validate_grid(self.grid)

    def on_city_paste(self, event):
        clipboard = wx.Clipboard.Get()
        if clipboard.Open():
            data_obj = wx.TextDataObject()
            if clipboard.GetData(data_obj):
                clipboard_data = data_obj.GetText()
                rows = [row.split('\t') for row in clipboard_data.split('\n') if row.strip()]

                if self.city_grid.GetNumberRows() > 0:
                    self.city_grid.DeleteRows(0, self.city_grid.GetNumberRows())

                if rows:
                    self.city_grid.AppendRows(len(rows))

                    for row_idx, row in enumerate(rows):
                        for col_idx, value in enumerate(row[:3]):  # Берем только первые 3 колонки
                            if row_idx < self.city_grid.GetNumberRows() and col_idx < self.city_grid.GetNumberCols():
                                self.city_grid.SetCellValue(row_idx, col_idx, value.strip())

                errors = DataValidator.validate_grid(self.city_grid)
                if errors:
                    wx.MessageBox(
                        f"Найдено {len(errors)} невалидных значений!\nПервая ошибка: {errors[0]}",
                        "Внимание", wx.OK | wx.ICON_WARNING
                    )

            clipboard.Close()

    def on_save(self, event):
        year = self.year_entry.GetValue().strip()
        if not year:
            wx.MessageBox("Введите год данных", "Ошибка", wx.OK | wx.ICON_ERROR)
            return

        data = []
        warnings = []
        errors = []

        for row in range(self.grid.GetNumberRows()):
            specialty = self.grid.GetCellValue(row, 0).strip()
            if not specialty:
                continue  # Пропускаем пустые строки

            values = []
            for col in range(1, 5):
                try:
                    value_str = self.grid.GetCellValue(row, col).strip()
                    if not value_str:  # Если ячейка пустая
                        values.append(0.0)
                        continue

                    value = float(value_str.replace(',', '.'))
                    if value < 0:
                        errors.append(f"Строка {row + 1}: Отрицательное значение в колонке {col + 1}")
                    values.append(value)
                except ValueError:
                    errors.append(f"Строка {row + 1}: Некорректное число в колонке {col + 1}")
                    values.append(0.0)  # Заменяем на 0 при ошибке

            # Проверка суммы мест (теперь как предупреждение, а не ошибка)
            total = sum(values)
            if total == 0:
                warnings.append(f"Строка {row + 1}: Общее количество мест равно нулю")

            data.append({
                'specialty': specialty,
                'budget': values[0],
                'target': values[1],
                'quota': values[2],
                'paid': values[3],
                'total': total
            })

        # Сначала показываем критические ошибки
        if errors:
            wx.MessageBox("Критические ошибки:\n" + "\n".join(errors[:5]) +
                          ("\n\n...и другие ошибки" if len(errors) > 5 else ""),
                          "Ошибки в данных", wx.OK | wx.ICON_ERROR)
            return

        # Затем предупреждения о нулевых значениях
        if warnings:
            response = wx.MessageBox(
                "Обнаружены специальности с нулевым количеством мест:\n" +
                "\n".join(warnings[:5]) +
                ("\n\n...и другие предупреждения" if len(warnings) > 5 else "") +
                "\n\nПродолжить сохранение?",
                "Внимание", wx.OK | wx.CANCEL | wx.ICON_WARNING)

            if response != wx.OK:
                return

        # Сохраняем данные
        self.main_frame.data[year] = pd.DataFrame(data)

        # Формируем итоговое сообщение
        message = f"Данные за {year} год сохранены!\n\n"
        message += f"Специальностей: {len(data)}\n"
        message += f"Всего мест: {sum(d['total'] for d in data)}\n"
        if warnings:
            message += f"\nПредупреждений: {len(warnings)}"

        wx.MessageBox(message, "Успех", wx.OK | wx.ICON_INFORMATION)

    def on_city_save(self, event):
        year = self.city_year_entry.GetValue().strip()
        if not year:
            wx.MessageBox("Введите год данных", "Ошибка", wx.OK | wx.ICON_ERROR)
            return

        data = []
        for row in range(self.city_grid.GetNumberRows()):
            city = self.city_grid.GetCellValue(row, 0)
            if not city:
                continue

            data.append({
                'city': city,
                'full_time': self.parse_number(self.city_grid.GetCellValue(row, 1)),
                'part_time': self.parse_number(self.city_grid.GetCellValue(row, 2))
            })

        if not hasattr(self.main_frame, 'city_data'):
            self.main_frame.city_data = {}

        self.main_frame.city_data[year] = pd.DataFrame(data)
        wx.MessageBox(f"Городские данные за {year} год сохранены!", "Успех", wx.OK | wx.ICON_INFORMATION)

    def on_clear(self, event):
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())

    def on_city_clear(self, event):
        if self.city_grid.GetNumberRows() > 0:
            self.city_grid.DeleteRows(0, self.city_grid.GetNumberRows())

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

    def load_city_data(self, clipboard_data):
        rows = [row.split('\t') for row in clipboard_data.split('\n') if row.strip()]
        self.city_grid.ClearGrid()
        if self.city_grid.GetNumberRows() > 0:
            self.city_grid.DeleteRows(0, self.city_grid.GetNumberRows())

        for row_idx, row in enumerate(rows):
            self.city_grid.AppendRows(1)
            for col_idx, value in enumerate(row[:3]):
                self.city_grid.SetCellValue(row_idx, col_idx, value.strip())

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

        # Добавляем переключатель типа истории
        self.history_type = wx.Choice(self, choices=["Все прогнозы", "По специальностям", "По городам"])
        self.history_type.SetSelection(0)
        self.history_type.Bind(wx.EVT_CHOICE, self.update_history_display)

        # Таблица истории
        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 5)  # Дата, Тип, Название, Прогноз, Статус
        self.setup_columns()

        vbox.Add(self.history_type, 0, wx.EXPAND | wx.ALL, 5)
        vbox.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(vbox)

    def setup_columns(self):
        columns = [
            ("Дата", 150),
            ("Тип данных", 120),
            ("Название", 250),
            ("Прогноз", 150),
            ("Статус", 100)
        ]
        for col, (label, width) in enumerate(columns):
            self.grid.SetColLabelValue(col, label)
            self.grid.SetColSize(col, width)

    def update_history_display(self, event=None):
        """Обновляет отображение истории в соответствии с выбранным фильтром"""
        if not hasattr(self.main_frame, 'history'):
            return

        self.grid.ClearGrid()
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())

        filter_type = self.history_type.GetSelection()  # 0-все, 1-специальности, 2-города

        row = 0
        for item in reversed(self.main_frame.history):
            # Применяем фильтр
            if filter_type == 1 and item['type'] != 'specialty':
                continue
            if filter_type == 2 and item['type'] != 'city':
                continue

            self.grid.AppendRows(1)
            self.grid.SetCellValue(row, 0, item['timestamp'])
            self.grid.SetCellValue(row, 1, "Специальности" if item['type'] == 'specialty' else "Города")
            self.grid.SetCellValue(row, 2, item['name'])
            self.grid.SetCellValue(row, 3, str(item['prediction']))
            self.grid.SetCellValue(row, 4, item['status'])

            # Подсветка строки в зависимости от статуса
            if item['status'] == "Ошибка":
                for col in range(5):
                    self.grid.SetCellBackgroundColour(row, col, wx.Colour(255, 200, 200))
            row += 1

        self.grid.AutoSizeColumns()
        self.grid.ForceRefresh()


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
• Ctrl+O - открыть проект"""

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
  - PNG (изображения графиков)"""

    def get_history_text(self):
        return """5. История:

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
        rows = grid.GetNumberRows()
        cols = grid.GetNumberCols()

        # Проверяем, что таблица не пустая
        if rows == 0 or cols == 0:
            return ["Таблица пуста"]

        for row in range(rows):
            for col in range(cols):
                # Для городских данных проверяем только числовые колонки (1 и 2)
                if col in (1, 2):
                    try:
                        value = grid.GetCellValue(row, col)
                        if not value.strip():  # Пустая строка
                            grid.SetCellBackgroundColour(row, col, wx.RED)
                            errors.append(f"Строка {row + 1}, Колонка {col + 1}: Пустое значение")
                            continue

                        # Пробуем преобразовать в число
                        float(value.replace(',', '.'))
                        grid.SetCellBackgroundColour(row, col, wx.WHITE)
                    except:
                        grid.SetCellBackgroundColour(row, col, wx.RED)
                        errors.append(f"Строка {row + 1}, Колонка {col + 1}: Некорректное число")
                else:
                    # Для нечисловых колонок просто проверяем, что не пустые
                    if not grid.GetCellValue(row, col).strip():
                        grid.SetCellBackgroundColour(row, col, wx.RED)
                        errors.append(f"Строка {row + 1}, Колонка {col + 1}: Пустое значение")
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