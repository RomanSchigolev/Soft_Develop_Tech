#!/usr/bin/env python

from PyQt5.QtWidgets import (QApplication, QMainWindow, QDialog)
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp
import numpy as np
from ui import (main, errorDialogForEmptyInputs, errorDialogOnCalculating, successDialog)
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


if __name__ == '__main__':
    app = QApplication([])
    main_app = MainApp()
    main_app.show()
    exit(app.exec_())
