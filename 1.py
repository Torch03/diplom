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


