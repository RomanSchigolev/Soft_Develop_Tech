# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'errorDialogOnCalculating.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 120)
        Dialog.setMinimumSize(QtCore.QSize(400, 120))
        Dialog.setMaximumSize(QtCore.QSize(400, 120))
        Dialog.setStyleSheet("background-color: rgb(230, 230, 230);")
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(10, 49, 381, 71))
        font = QtGui.QFont()
        font.setFamily("Avenir")
        font.setPointSize(11)
        self.textBrowser.setFont(font)
        self.textBrowser.setStyleSheet("border: none;")
        self.textBrowser.setObjectName("textBrowser")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(160, 20, 81, 16))
        font = QtGui.QFont()
        font.setFamily("Avenir Heavy")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Ошибка"))
        self.textBrowser.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Avenir\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\';\">Данные матрицы не являются согласованными. Количество столбцов в первой матрице должно совпадать с количеством строк во второй матрице.</span></p></body></html>"))
        self.label.setText(_translate("Dialog", "Ошибка!"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())