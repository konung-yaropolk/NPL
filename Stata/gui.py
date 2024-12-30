import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, QMessageBox
from PyQt6.QtCore import Qt
import numpy as np
from stats import StatisticalAnalysis  # Replace 'your_module' with the actual module name

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Statistical Analysis Tool")
        self.setGeometry(100, 100, 600, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        self.input_label = QLabel("Enter Data (comma-separated):")
        layout.addWidget(self.input_label)

        self.input_field = QLineEdit()
        layout.addWidget(self.input_field)

        self.result_label = QLabel("Results:")
        layout.addWidget(self.result_label)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        layout.addWidget(self.result_display)

        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_button)

        central_widget.setLayout(layout)

    def run_analysis(self):
        data_text = self.input_field.text()
        try:
            groups_list = [list(map(float, group.split(','))) for group in data_text.split(';')]
            analysis = StatisticalAnalysis(groups_list)
            analysis.RunAuto()
            result = analysis.GetSummary()
            self.result_display.setPlainText(result)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
