#!/usr/bin/env python

from PyQt5.QtWidgets import (QApplication, QMainWindow, QDialog)
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp
import numpy as np
from ui import (main, errorDialogForEmptyInputs,
                errorDialogOnCalculating, successDialog)
from os import (getcwd, mkdir, path)


class SuccessDialog(successDialog.Ui_Dialog, QDialog):
    '''Окно, подтверждающее успех процедуры'''

    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)


class ErrorDialogForEmptyInputs(errorDialogForEmptyInputs.Ui_Dialog, QDialog):
    '''Окно с ошибкой при пустых инпутах'''

    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)


class ErrorDialogOnCalculating(errorDialogOnCalculating.Ui_Dialog, QDialog):
    '''Окно с ошибкой при расчете'''

    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)


class MainApp(main.Ui_MainWindow, QMainWindow):
    '''Главное окно приложения'''

    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        # Создаем регулярное выражение и устанавливаем валидатор для полей ввода
        regexp_query = QRegExp('[^0|\D]\d*$')
        input_validator = QRegExpValidator(regexp_query)
        input_validator.setRegExp(regexp_query)

        self.rows_A.setValidator(input_validator)
        self.cols_A.setValidator(input_validator)
        self.rows_B.setValidator(input_validator)
        self.cols_B.setValidator(input_validator)

        self.generate_and_calculate.clicked.connect(
            self.calculate_product_of_matrices)

    # Действия по нажатию кнопки "Сгенерировать и рассчитать"
    def calculate_product_of_matrices(self):
        # Заполнены ли поля для ввода
        is_input_filled_in = self.checking_filling_in_of_the_input()

        if is_input_filled_in:
            rows_A = int(self.rows_A.text())
            cols_A = int(self.cols_A.text())
            rows_B = int(self.rows_B.text())
            cols_B = int(self.cols_B.text())

            # Проверка на согласованность матриц
            if cols_A != rows_B:
                error_dialog_on_calculating = ErrorDialogOnCalculating()
                error_dialog_on_calculating.show()
                error_dialog_on_calculating.exec_()
            else:
                # Заполняем матрицы случайными целыми числами от 0 до 100
                first_matrix_A = np.random.randint(0, 100, (rows_A, cols_A))
                second_matrix_B = np.random.randint(0, 100, (rows_B, cols_B))
                result_matrix_C = np.dot(first_matrix_A, second_matrix_B)

                MainApp.__generate_csv_files(
                    first_matrix_A, second_matrix_B, result_matrix_C)

                success_dialog = SuccessDialog()
                success_dialog.show()
                success_dialog.exec_()
        else:
            error_dialog_for_empty_inputs = ErrorDialogForEmptyInputs()
            error_dialog_for_empty_inputs.show()
            error_dialog_for_empty_inputs.exec_()

    # Проверка на заполнение полей ввода
    def checking_filling_in_of_the_input(self):
        if not (self.rows_A.text()) or \
                not (self.cols_A.text()) or \
                not (self.rows_B.text()) or \
                not (self.cols_B.text()):
            return False
        return True

    @staticmethod
    # Генерируем .csv файлы для каждой матрицы
    def __generate_csv_files(first_matrix, second_matrix, result_matrix):
        current_path = getcwd()
        folder_for_csv_files = path.join(current_path, 'csv')
        if not path.exists(folder_for_csv_files):
            mkdir(folder_for_csv_files)

        np.savetxt(
            path.join(folder_for_csv_files, 'first_matrix_A.csv'),
            first_matrix,
            delimiter=';',
            fmt='%d'
        )

        np.savetxt(
            path.join(folder_for_csv_files, 'second_matrix_B.csv'),
            second_matrix,
            delimiter=';',
            fmt='%d'
        )

        np.savetxt(
            path.join(folder_for_csv_files, 'result_matrix_C.csv'),
            result_matrix,
            delimiter=';',
            fmt='%d'
        )


if __name__ == '__main__':
    app = QApplication([])
    main_app = MainApp()
    main_app.show()
    exit(app.exec_())
