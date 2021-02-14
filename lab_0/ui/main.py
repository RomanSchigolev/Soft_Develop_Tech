# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(661, 300)
        MainWindow.setMinimumSize(QtCore.QSize(661, 300))
        MainWindow.setMaximumSize(QtCore.QSize(661, 300))
        MainWindow.setStyleSheet("background-color: rgb(230, 230, 230);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.generate_and_calculate = QtWidgets.QPushButton(self.centralwidget)
        self.generate_and_calculate.setGeometry(QtCore.QRect(245, 200, 170, 30))
        font = QtGui.QFont()
        font.setFamily("Avenir")
        font.setPointSize(9)
        self.generate_and_calculate.setFont(font)
        self.generate_and_calculate.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.generate_and_calculate.setStyleSheet("background-color: rgb(69, 128, 255);\n"
"color: #ffffff;")
        self.generate_and_calculate.setObjectName("generate_and_calculate")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 0, 621, 161))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(100)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Avenir")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_5.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setContentsMargins(-1, -1, -1, 0)
        self.verticalLayout_11.setSpacing(20)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_2.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout_2.setSpacing(5)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.rows_title_A = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Avenir")
        font.setPointSize(10)
        self.rows_title_A.setFont(font)
        self.rows_title_A.setObjectName("rows_title_A")
        self.verticalLayout_2.addWidget(self.rows_title_A)
        self.rows_A = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rows_A.sizePolicy().hasHeightForWidth())
        self.rows_A.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Avenir")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.rows_A.setFont(font)
        self.rows_A.setStyleSheet("")
        self.rows_A.setMaxVisibleItems(10)
        self.rows_A.setObjectName("rows_A")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.rows_A.addItem("")
        self.verticalLayout_2.addWidget(self.rows_A)
        self.verticalLayout_11.addLayout(self.verticalLayout_2)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setSpacing(5)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.cols_title_A = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Avenir")
        font.setPointSize(10)
        self.cols_title_A.setFont(font)
        self.cols_title_A.setObjectName("cols_title_A")
        self.verticalLayout_10.addWidget(self.cols_title_A)
        self.cols_A = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cols_A.sizePolicy().hasHeightForWidth())
        self.cols_A.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Avenir")
        font.setPointSize(9)
        self.cols_A.setFont(font)
        self.cols_A.setMaxVisibleItems(10)
        self.cols_A.setObjectName("cols_A")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.cols_A.addItem("")
        self.verticalLayout_10.addWidget(self.cols_A)
        self.verticalLayout_11.addLayout(self.verticalLayout_10)
        self.verticalLayout_5.addLayout(self.verticalLayout_11)
        self.verticalLayout.addLayout(self.verticalLayout_5)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Avenir")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_6.addWidget(self.label_4)
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setSpacing(0)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setSpacing(20)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_9.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout_9.setSpacing(5)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.rows_title_B = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Avenir")
        font.setPointSize(10)
        self.rows_title_B.setFont(font)
        self.rows_title_B.setObjectName("rows_title_B")
        self.verticalLayout_9.addWidget(self.rows_title_B)
        self.rows_B = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rows_B.sizePolicy().hasHeightForWidth())
        self.rows_B.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Avenir")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.rows_B.setFont(font)
        self.rows_B.setMaxVisibleItems(10)
        self.rows_B.setObjectName("rows_B")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.rows_B.addItem("")
        self.verticalLayout_9.addWidget(self.rows_B)
        self.verticalLayout_12.addLayout(self.verticalLayout_9)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_8.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout_8.setSpacing(5)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.cols_title_B = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Avenir")
        font.setPointSize(10)
        self.cols_title_B.setFont(font)
        self.cols_title_B.setObjectName("cols_title_B")
        self.verticalLayout_8.addWidget(self.cols_title_B)
        self.cols_B = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cols_B.sizePolicy().hasHeightForWidth())
        self.cols_B.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Avenir")
        font.setPointSize(9)
        self.cols_B.setFont(font)
        self.cols_B.setMaxVisibleItems(10)
        self.cols_B.setObjectName("cols_B")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.cols_B.addItem("")
        self.verticalLayout_8.addWidget(self.cols_B)
        self.verticalLayout_12.addLayout(self.verticalLayout_8)
        self.verticalLayout_13.addLayout(self.verticalLayout_12)
        self.verticalLayout_6.addLayout(self.verticalLayout_13)
        self.horizontalLayout.addLayout(self.verticalLayout_6)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Произведение Матриц"))
        self.generate_and_calculate.setText(_translate("MainWindow", "Сгенерировать и рассчитать"))
        self.label.setText(_translate("MainWindow", "Размерность первой матрицы:"))
        self.rows_title_A.setText(_translate("MainWindow", "Кол-во строк:"))
        self.rows_A.setItemText(0, _translate("MainWindow", "2"))
        self.rows_A.setItemText(1, _translate("MainWindow", "3"))
        self.rows_A.setItemText(2, _translate("MainWindow", "4"))
        self.rows_A.setItemText(3, _translate("MainWindow", "5"))
        self.rows_A.setItemText(4, _translate("MainWindow", "6"))
        self.rows_A.setItemText(5, _translate("MainWindow", "7"))
        self.rows_A.setItemText(6, _translate("MainWindow", "8"))
        self.rows_A.setItemText(7, _translate("MainWindow", "9"))
        self.rows_A.setItemText(8, _translate("MainWindow", "10"))
        self.rows_A.setItemText(9, _translate("MainWindow", "11"))
        self.rows_A.setItemText(10, _translate("MainWindow", "12"))
        self.rows_A.setItemText(11, _translate("MainWindow", "13"))
        self.rows_A.setItemText(12, _translate("MainWindow", "14"))
        self.rows_A.setItemText(13, _translate("MainWindow", "15"))
        self.rows_A.setItemText(14, _translate("MainWindow", "16"))
        self.rows_A.setItemText(15, _translate("MainWindow", "17"))
        self.rows_A.setItemText(16, _translate("MainWindow", "18"))
        self.rows_A.setItemText(17, _translate("MainWindow", "19"))
        self.rows_A.setItemText(18, _translate("MainWindow", "20"))
        self.rows_A.setItemText(19, _translate("MainWindow", "21"))
        self.rows_A.setItemText(20, _translate("MainWindow", "22"))
        self.rows_A.setItemText(21, _translate("MainWindow", "23"))
        self.rows_A.setItemText(22, _translate("MainWindow", "24"))
        self.rows_A.setItemText(23, _translate("MainWindow", "25"))
        self.rows_A.setItemText(24, _translate("MainWindow", "26"))
        self.rows_A.setItemText(25, _translate("MainWindow", "27"))
        self.rows_A.setItemText(26, _translate("MainWindow", "28"))
        self.rows_A.setItemText(27, _translate("MainWindow", "29"))
        self.rows_A.setItemText(28, _translate("MainWindow", "30"))
        self.cols_title_A.setText(_translate("MainWindow", "Кол-во столбцов:"))
        self.cols_A.setItemText(0, _translate("MainWindow", "2"))
        self.cols_A.setItemText(1, _translate("MainWindow", "3"))
        self.cols_A.setItemText(2, _translate("MainWindow", "4"))
        self.cols_A.setItemText(3, _translate("MainWindow", "5"))
        self.cols_A.setItemText(4, _translate("MainWindow", "6"))
        self.cols_A.setItemText(5, _translate("MainWindow", "7"))
        self.cols_A.setItemText(6, _translate("MainWindow", "8"))
        self.cols_A.setItemText(7, _translate("MainWindow", "9"))
        self.cols_A.setItemText(8, _translate("MainWindow", "10"))
        self.cols_A.setItemText(9, _translate("MainWindow", "11"))
        self.cols_A.setItemText(10, _translate("MainWindow", "12"))
        self.cols_A.setItemText(11, _translate("MainWindow", "13"))
        self.cols_A.setItemText(12, _translate("MainWindow", "14"))
        self.cols_A.setItemText(13, _translate("MainWindow", "15"))
        self.cols_A.setItemText(14, _translate("MainWindow", "16"))
        self.cols_A.setItemText(15, _translate("MainWindow", "17"))
        self.cols_A.setItemText(16, _translate("MainWindow", "18"))
        self.cols_A.setItemText(17, _translate("MainWindow", "19"))
        self.cols_A.setItemText(18, _translate("MainWindow", "20"))
        self.cols_A.setItemText(19, _translate("MainWindow", "21"))
        self.cols_A.setItemText(20, _translate("MainWindow", "22"))
        self.cols_A.setItemText(21, _translate("MainWindow", "23"))
        self.cols_A.setItemText(22, _translate("MainWindow", "24"))
        self.cols_A.setItemText(23, _translate("MainWindow", "25"))
        self.cols_A.setItemText(24, _translate("MainWindow", "26"))
        self.cols_A.setItemText(25, _translate("MainWindow", "27"))
        self.cols_A.setItemText(26, _translate("MainWindow", "28"))
        self.cols_A.setItemText(27, _translate("MainWindow", "29"))
        self.cols_A.setItemText(28, _translate("MainWindow", "30"))
        self.label_4.setText(_translate("MainWindow", "Размерность второй матрицы:"))
        self.rows_title_B.setText(_translate("MainWindow", "Кол-во строк:"))
        self.rows_B.setItemText(0, _translate("MainWindow", "2"))
        self.rows_B.setItemText(1, _translate("MainWindow", "3"))
        self.rows_B.setItemText(2, _translate("MainWindow", "4"))
        self.rows_B.setItemText(3, _translate("MainWindow", "5"))
        self.rows_B.setItemText(4, _translate("MainWindow", "6"))
        self.rows_B.setItemText(5, _translate("MainWindow", "7"))
        self.rows_B.setItemText(6, _translate("MainWindow", "8"))
        self.rows_B.setItemText(7, _translate("MainWindow", "9"))
        self.rows_B.setItemText(8, _translate("MainWindow", "10"))
        self.rows_B.setItemText(9, _translate("MainWindow", "11"))
        self.rows_B.setItemText(10, _translate("MainWindow", "12"))
        self.rows_B.setItemText(11, _translate("MainWindow", "13"))
        self.rows_B.setItemText(12, _translate("MainWindow", "14"))
        self.rows_B.setItemText(13, _translate("MainWindow", "15"))
        self.rows_B.setItemText(14, _translate("MainWindow", "16"))
        self.rows_B.setItemText(15, _translate("MainWindow", "17"))
        self.rows_B.setItemText(16, _translate("MainWindow", "18"))
        self.rows_B.setItemText(17, _translate("MainWindow", "19"))
        self.rows_B.setItemText(18, _translate("MainWindow", "20"))
        self.rows_B.setItemText(19, _translate("MainWindow", "21"))
        self.rows_B.setItemText(20, _translate("MainWindow", "22"))
        self.rows_B.setItemText(21, _translate("MainWindow", "23"))
        self.rows_B.setItemText(22, _translate("MainWindow", "24"))
        self.rows_B.setItemText(23, _translate("MainWindow", "25"))
        self.rows_B.setItemText(24, _translate("MainWindow", "26"))
        self.rows_B.setItemText(25, _translate("MainWindow", "27"))
        self.rows_B.setItemText(26, _translate("MainWindow", "28"))
        self.rows_B.setItemText(27, _translate("MainWindow", "29"))
        self.rows_B.setItemText(28, _translate("MainWindow", "30"))
        self.cols_title_B.setText(_translate("MainWindow", "Кол-во столбцов:"))
        self.cols_B.setItemText(0, _translate("MainWindow", "2"))
        self.cols_B.setItemText(1, _translate("MainWindow", "3"))
        self.cols_B.setItemText(2, _translate("MainWindow", "4"))
        self.cols_B.setItemText(3, _translate("MainWindow", "5"))
        self.cols_B.setItemText(4, _translate("MainWindow", "6"))
        self.cols_B.setItemText(5, _translate("MainWindow", "7"))
        self.cols_B.setItemText(6, _translate("MainWindow", "8"))
        self.cols_B.setItemText(7, _translate("MainWindow", "9"))
        self.cols_B.setItemText(8, _translate("MainWindow", "10"))
        self.cols_B.setItemText(9, _translate("MainWindow", "11"))
        self.cols_B.setItemText(10, _translate("MainWindow", "12"))
        self.cols_B.setItemText(11, _translate("MainWindow", "13"))
        self.cols_B.setItemText(12, _translate("MainWindow", "14"))
        self.cols_B.setItemText(13, _translate("MainWindow", "15"))
        self.cols_B.setItemText(14, _translate("MainWindow", "16"))
        self.cols_B.setItemText(15, _translate("MainWindow", "17"))
        self.cols_B.setItemText(16, _translate("MainWindow", "18"))
        self.cols_B.setItemText(17, _translate("MainWindow", "19"))
        self.cols_B.setItemText(18, _translate("MainWindow", "20"))
        self.cols_B.setItemText(19, _translate("MainWindow", "21"))
        self.cols_B.setItemText(20, _translate("MainWindow", "22"))
        self.cols_B.setItemText(21, _translate("MainWindow", "23"))
        self.cols_B.setItemText(22, _translate("MainWindow", "24"))
        self.cols_B.setItemText(23, _translate("MainWindow", "25"))
        self.cols_B.setItemText(24, _translate("MainWindow", "26"))
        self.cols_B.setItemText(25, _translate("MainWindow", "27"))
        self.cols_B.setItemText(26, _translate("MainWindow", "28"))
        self.cols_B.setItemText(27, _translate("MainWindow", "29"))
        self.cols_B.setItemText(28, _translate("MainWindow", "30"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
