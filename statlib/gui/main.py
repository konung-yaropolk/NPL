import sys
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTextEdit, QPushButton, QPlainTextEdit

import statlib
from mainwindow_layout import Ui_MainWindow


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.load_UI_from_py()
        # self.load_UI_from_ui()

    # def load_UI_from_ui(self,):

    #     # Load the .ui file
    #     uic.loadUi('mainwindow_layout.ui', self)

    #     # Find UI elements
    #     self.input_field = self.findChild(QPlainTextEdit, 'input_field')
    #     self.result_display = self.findChild(QTextEdit, 'result_display')
    #     self.runAutoButton = self.findChild(QPushButton, 'runAutoButton')

    #     # Connect button to function
    #     self.runAutoButton.clicked.connect(self.run_analysis)

    def load_UI_from_py(self,):

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Find UI elements
        self.input_field = self.ui.input_field
        self.result_display = self.ui.result_display
        self.runAutoButton = self.ui.runAutoButton
        self.runButton = self.ui.runButton
        self.dependendGroups = self.ui.dependendGroups
        self.oneTailed = self.ui.oneTailed
        self.twoTailed = self.ui.twoTailed
        self.popMean = self.ui.popMean
        self.t_test_single_sample = self.ui.t_test_single_sample
        self.wilcoxon_single_sample = self.ui.wilcoxon_single_sample
        self.t_test_paired = self.ui.t_test_paired
        self.wilcoxon = self.ui.wilcoxon
        self.t_test_independend = self.ui.t_test_independend
        self.mann_whitney = self.ui.mann_whitney
        self.friedman = self.ui.friedman
        self.anova = self.ui.anova
        self.kruskal_wallis = self.ui.kruskal_wallis
        self.dependendGroups = self.ui.dependendGroups
        self.errorMsg = self.ui.errorMsg

        # Connect button to function
        self.runAutoButton.clicked.connect(self.runAuto)
        self.runButton.clicked.connect(self.runManual)

    def listify_text(self, data_text):
        matrix = [list(group.split('\t')) for group in data_text.split('\n')]
        max_len = max(len(row) for row in matrix)
        padded_matrix = [row + [None] * (max_len - len(row)) for row in matrix]
        transposed = [[padded_matrix[j][i]
                       for j in range(len(padded_matrix))] for i in range(max_len)]

        # Remove None values if padding was used
        return [[element for element in row if element is not None] for row in transposed]

    def runAuto(self):
        self.errorMsg.setText('')
        data_text = self.input_field.toPlainText()
        # try:
        groups_list = self.listify_text(data_text)
        analysis = statlib.StatisticalAnalysis(
            groups_list,
            paired=self.dependendGroups.isChecked(),
            tails=2 if self.twoTailed.isChecked() else 1,
            popmean=float(self.popMean.text()))
        analysis.RunAuto()
        result = analysis.GetSummary()
        self.result_display.setPlainText(result)
        # except Exception as e:
        #     e = 'Error: \n' + str(e)
        #     print(e)
        #     self.errorMsg.setText(e)
        #     # QMessageBox.critical(self, "Error", str(e))

    def runManual(self):
        self.errorMsg.setText('')
        data_text = self.input_field.toPlainText()
        try:
            groups_list = self.listify_text(data_text)
            analysis = statlib.StatisticalAnalysis(
                groups_list,
                paired=self.dependendGroups.isChecked(),
                tails=2 if self.twoTailed.isChecked() else 1,
                popmean=float(self.popMean.text()))

            checkbuttons_state_list = [
                self.anova.isChecked(),
                self.friedman.isChecked(),
                self.kruskal_wallis.isChecked(),
                self.mann_whitney.isChecked(),
                self.t_test_independend.isChecked(),
                self.t_test_paired.isChecked(),
                self.t_test_single_sample.isChecked(),
                self.wilcoxon.isChecked(),
                self.wilcoxon_single_sample.isChecked(),
            ]

            checked_test_ids = []
            for i in range(len(analysis.test_ids_all)):
                if checkbuttons_state_list[i]:
                    checked_test_ids.append(analysis.test_ids_all[i])

            for test_id in checked_test_ids:
                analysis.RunManual(test_id)

            result = analysis.GetSummary()
            self.result_display.setPlainText(result)
        except Exception as e:
            e = 'Error: \n' + str(e)
            print(e)
            self.errorMsg.setText(e)
            # QMessageBox.critical(self, "Error", str(e))


if __name__ == '__main__':
    # import pyi_splash
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    # pyi_splash.close()
    sys.exit(app.exec())
