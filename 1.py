import wx
import wx.grid as gridlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('WXAgg')  # Критически важно для macOS
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from sklearn.metrics import r2_score


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

    def train(self, X, y, epochs=2000, lr=0.0001, batch_size=64):
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


class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="University Analytics", size=(1200, 800))
        self.nn = None
        self.data = {}

        # Настройки для macOS
        self.SetSizeHints(800, 600)
        self.init_ui()
        self.Centre()
        self.Show(True)  # Явное отображение окна

    def init_ui(self):
        panel = wx.Panel(self)
        notebook = wx.Notebook(panel)

        # Вкладка ввода данных
        self.input_tab = InputTab(notebook, self)
        notebook.AddPage(self.input_tab, "Data Input")

        # Вкладка прогноза
        self.prediction_tab = PredictionTab(notebook, self)
        notebook.AddPage(self.prediction_tab, "Prediction")

        # Статус бар
        self.CreateStatusBar()
        self.SetStatusText("Ready")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 5)
        panel.SetSizerAndFit(sizer)
        self.Layout()

    def prepare_training_data(self):
        features = []
        targets = []
        years = sorted(self.data.keys())

        if len(years) < 2:
            raise ValueError("Need at least 2 years of data")

        for i in range(len(years) - 1):
            current = self.data[years[i]].copy()
            next_year = self.data[years[i + 1]].copy()

            current['specialty'] = current['specialty'].str.strip().str.lower()
            next_year['specialty'] = next_year['specialty'].str.strip().str.lower()

            merged = pd.merge(
                current,
                next_year,
                on='specialty',
                how='inner',
                suffixes=('_current', '_next'))

            if merged.empty:
                raise ValueError(f"No matching specialties between {years[i]} and {years[i + 1]}")

            if len(merged) < 5:
                wx.MessageBox(
                    f"Warning: Only {len(merged)} matching specialties between {years[i]} and {years[i + 1]}",
                    "Data Warning",
                    wx.OK | wx.ICON_WARNING
                )

            features.append(merged[['budget_current', 'target_current',
                                    'quota_current', 'paid_current']].values)
            targets.append(merged[['budget_next', 'target_next',
                                   'quota_next', 'paid_next']].values)

        return np.vstack(features), np.vstack(targets)


class InputTab(wx.Panel):
    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Панель управления
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.year_entry = wx.TextCtrl(self, size=(120, -1))
        hbox.Add(wx.StaticText(self, label="Year:"), 0, wx.ALIGN_CENTER | wx.ALL, 5)
        hbox.Add(self.year_entry, 0, wx.ALL, 5)

        # Кнопки с macOS-совместимыми размерами
        buttons = [
            ("Paste from Excel", self.on_paste),
            ("Save Data", self.on_save),
            ("Clear Table", self.on_clear)
        ]

        for label, handler in buttons:
            btn = wx.Button(self, label=label, size=(140, 28))
            btn.Bind(wx.EVT_BUTTON, handler)
            hbox.Add(btn, 0, wx.ALL, 5)

        vbox.Add(hbox, 0, wx.EXPAND)

        # Таблица с настройками для macOS
        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 5)
        self.grid.SetColLabelValue(0, "Specialty")
        self.grid.SetColLabelValue(1, "Budget")
        self.grid.SetColLabelValue(2, "Target")
        self.grid.SetColLabelValue(3, "Quota")
        self.grid.SetColLabelValue(4, "Paid")

        # Настройки внешнего вида
        font = wx.Font(12, wx.FONTFAMILY_DEFAULT,
                       wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.grid.SetLabelFont(font)
        self.grid.SetDefaultCellFont(font)
        self.grid.AutoSizeColumns()

        vbox.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(vbox)

    def on_paste(self, event):
        clipboard = wx.Clipboard.Get()
        if clipboard.Open():
            data_obj = wx.TextDataObject()
            if clipboard.GetData(data_obj):
                self.load_data(data_obj.GetText())
            clipboard.Close()

    def load_data(self, clipboard_data):
        self.grid.ClearGrid()
        rows = [row.split('\t') for row in clipboard_data.split('\n') if row.strip()]

        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())

        for row_idx, row in enumerate(rows):
            self.grid.AppendRows(1)
            for col_idx, value in enumerate(row[:5]):
                self.grid.SetCellValue(row_idx, col_idx, value.strip())

        self.grid.AutoSizeColumns()
        self.main_frame.SetStatusText(f"Loaded {len(rows)} records")

    def on_save(self, event):
        year = self.year_entry.GetValue().strip()
        if not year:
            wx.MessageBox("Please enter a year", "Error", wx.OK | wx.ICON_ERROR)
            return

        data = []
        for row in range(self.grid.GetNumberRows()):
            data.append({
                'specialty': self.grid.GetCellValue(row, 0),
                'budget': self.parse_number(self.grid.GetCellValue(row, 1)),
                'target': self.parse_number(self.grid.GetCellValue(row, 2)),
                'quota': self.parse_number(self.grid.GetCellValue(row, 3)),
                'paid': self.parse_number(self.grid.GetCellValue(row, 4))
            })

        self.main_frame.data[year] = pd.DataFrame(data)
        wx.MessageBox(f"Data for {year} saved successfully!", "Success", wx.OK | wx.ICON_INFORMATION)

    def parse_number(self, value):
        try:
            return float(str(value).replace(',', '.').replace(' ', ''))
        except:
            return 0.0

    def on_clear(self, event):
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())


class PredictionTab(wx.Panel):
    def __init__(self, parent, main_frame):
        super().__init__(parent)
        self.main_frame = main_frame
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Кнопки с macOS-стилем
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.train_btn = wx.Button(self, label="Train Model", size=(120, 28))
        self.train_btn.Bind(wx.EVT_BUTTON, self.on_train)
        hbox.Add(self.train_btn, 0, wx.ALL, 5)

        self.predict_btn = wx.Button(self, label="Generate Prediction", size=(160, 28))
        self.predict_btn.Bind(wx.EVT_BUTTON, self.on_predict)
        hbox.Add(self.predict_btn, 0, wx.ALL, 5)
        vbox.Add(hbox, 0, wx.EXPAND)

        # Текстовое поле вывода
        self.output = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY)
        vbox.Add(self.output, 1, wx.EXPAND | wx.ALL, 5)

        # Область графика
        vbox.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(vbox)

    def on_train(self, event):
        try:
            X, y = self.main_frame.prepare_training_data()
            self.main_frame.nn = NeuralNetwork(X.shape[1])
            self.main_frame.nn.train(X, y)
            self.plot_results(X, y)
            wx.MessageBox("Model trained successfully!", "Success", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)

    def on_predict(self, event):
        if not self.main_frame.nn or not self.main_frame.data:
            wx.MessageBox("Please train the model first", "Error", wx.OK | wx.ICON_ERROR)
            return

        try:
            latest_year = sorted(self.main_frame.data.keys())[-1]
            df = self.main_frame.data[latest_year]
            X = df[['budget', 'target', 'quota', 'paid']].values
            pred = self.main_frame.nn.predict(X)

            result = f"Prediction for {int(latest_year) + 1}:\n"
            result += "-" * 80 + "\n"
            for i, row in df.iterrows():
                result += f"{row['specialty'][:40]:<40} | "
                result += " | ".join([f"{v:>10.1f}" for v in pred[i]]) + "\n"

            self.output.SetValue(result)
        except Exception as e:
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)

    def plot_results(self, X, y):
        ax = self.figure.add_subplot(111)
        ax.clear()

        pred = self.main_frame.nn.predict(X)
        pred = pred.flatten()
        y_true = (y * self.main_frame.nn.y_std + self.main_frame.nn.y_mean).flatten()

        ax.scatter(y_true, pred, alpha=0.3)
        ax.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()], 'r--')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predictions")
        ax.set_title(f"R2 Score: {r2_score(y_true, pred):.2f}")
        self.canvas.draw()


if __name__ == "__main__":
    # Настройки для macOS
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()