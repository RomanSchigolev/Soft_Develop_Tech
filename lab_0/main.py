from PyQt5.QtWidgets import (QApplication, QMainWindow, QDialog)
import numpy as np
from ui import (main, errorDialog, successDialog)
from os import getcwd, mkdir, path


class SuccessDialog(successDialog.Ui_Dialog, QDialog):
    '''Окно, подтверждающее успех процедуры'''

    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)


class ErrorDialog(errorDialog.Ui_Dialog, QDialog):
    '''Окно, поясняющее ошибку'''

    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)


class MainApp(main.Ui_MainWindow, QMainWindow):
    '''Главное окно приложения'''

    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        self.generate_and_calculate.clicked.connect(self.calculate_product_of_matrices)

    # Действия по нажатию кнопки "Сгенерировать и рассчитать"
    def calculate_product_of_matrices(self):
        rows_A = int(self.rows_A.currentText())
        cols_A = int(self.cols_A.currentText())
        rows_B = int(self.rows_B.currentText())
        cols_B = int(self.cols_B.currentText())

        # Проверка на согласованность матриц
        if cols_A != rows_B:
            error_dialog = ErrorDialog()
            error_dialog.show()
            error_dialog.exec_()
        else:
            # Заполняем матрицы случайными целыми числами от 0 до 50
            first_matrix_A = np.random.randint(0, 50, (rows_A, cols_A))
            second_matrix_B = np.random.randint(0, 50, (rows_B, cols_B))
            result_matrix_C = np.dot(first_matrix_A, second_matrix_B)

            self.generate_csv_files(first_matrix_A, second_matrix_B, result_matrix_C)

            success_dialog = SuccessDialog()
            success_dialog.show()
            success_dialog.exec_()

    @staticmethod
    # Генерируем .csv файлы для каждой матрицы
    def generate_csv_files(first_matrix, second_matrix, result_matrix):
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
