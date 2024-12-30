import sys
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTextEdit, QPushButton, QPlainTextEdit

# Replace 'your_module' with the actual module name where your StatisticalAnalysis class is defined
from stats import StatisticalAnalysis

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # Load the .ui file
        uic.loadUi('mainwindow.ui', self)

        # Find UI elements
        self.input_field = self.findChild(QPlainTextEdit, 'input_field')
        self.result_display = self.findChild(QTextEdit, 'result_display')
        self.run_button = self.findChild(QPushButton, 'runButton')

        # Connect button to function
        self.run_button.clicked.connect(self.run_analysis)

    def transpose_matrix(self, matrix):
        max_len = max(len(row) for row in matrix)
        padded_matrix = [row + [None] * (max_len - len(row)) for row in matrix]
        transposed = [[padded_matrix[j][i] for j in range(len(padded_matrix))] for i in range(max_len)]
        # Remove None values if padding was used
        return [[element for element in row if element is not None] for row in transposed]

    def run_analysis(self):
        data_text = self.input_field.toPlainText()
        try:
            groups_list = self.transpose_matrix([list( group.split('\t')) for group in data_text.split('\n')])
            analysis = StatisticalAnalysis(groups_list)
            analysis.RunAuto()
            result = analysis.GetSummary()
            self.result_display.setPlainText(result)
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Error", str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
