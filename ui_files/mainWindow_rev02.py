# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindow_rev02.ui'
#
# Created: Thu Apr 30 15:25:13 2015
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName(_fromUtf8("mainWindow"))
        mainWindow.resize(575, 350)
        mainWindow.setMinimumSize(QtCore.QSize(575, 350))
        mainWindow.setMaximumSize(QtCore.QSize(575, 350))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icon36x36.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        mainWindow.setWindowIcon(icon)
        self.centralwidget = QtGui.QWidget(mainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.layoutWidget_2 = QtGui.QWidget(self.centralwidget)
        self.layoutWidget_2.setGeometry(QtCore.QRect(390, 10, 171, 301))
        self.layoutWidget_2.setObjectName(_fromUtf8("layoutWidget_2"))
        self.PlotsverticalLayout = QtGui.QVBoxLayout(self.layoutWidget_2)
        self.PlotsverticalLayout.setMargin(0)
        self.PlotsverticalLayout.setObjectName(_fromUtf8("PlotsverticalLayout"))
        self.desiredPlotslabel = QtGui.QLabel(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.desiredPlotslabel.setFont(font)
        self.desiredPlotslabel.setAlignment(QtCore.Qt.AlignCenter)
        self.desiredPlotslabel.setObjectName(_fromUtf8("desiredPlotslabel"))
        self.PlotsverticalLayout.addWidget(self.desiredPlotslabel)
        self.heatMapCheckbox = QtGui.QCheckBox(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.heatMapCheckbox.setFont(font)
        self.heatMapCheckbox.setChecked(True)
        self.heatMapCheckbox.setObjectName(_fromUtf8("heatMapCheckbox"))
        self.PlotsverticalLayout.addWidget(self.heatMapCheckbox)
        self.contourPlotCheckbox = QtGui.QCheckBox(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.contourPlotCheckbox.setFont(font)
        self.contourPlotCheckbox.setObjectName(_fromUtf8("contourPlotCheckbox"))
        self.PlotsverticalLayout.addWidget(self.contourPlotCheckbox)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem = QtGui.QSpacerItem(50, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.contourLinesLabel = QtGui.QLabel(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.contourLinesLabel.setFont(font)
        self.contourLinesLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.contourLinesLabel.setObjectName(_fromUtf8("contourLinesLabel"))
        self.horizontalLayout.addWidget(self.contourLinesLabel)
        self.contourLinesSpinBox = QtGui.QSpinBox(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.contourLinesSpinBox.setFont(font)
        self.contourLinesSpinBox.setProperty("value", 10)
        self.contourLinesSpinBox.setObjectName(_fromUtf8("contourLinesSpinBox"))
        self.horizontalLayout.addWidget(self.contourLinesSpinBox)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.PlotsverticalLayout.addLayout(self.horizontalLayout)
        self.uniformityPlotCheckbox = QtGui.QCheckBox(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.uniformityPlotCheckbox.setFont(font)
        self.uniformityPlotCheckbox.setObjectName(_fromUtf8("uniformityPlotCheckbox"))
        self.PlotsverticalLayout.addWidget(self.uniformityPlotCheckbox)
        self.longAxisPlotCheckbox = QtGui.QCheckBox(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.longAxisPlotCheckbox.setFont(font)
        self.longAxisPlotCheckbox.setObjectName(_fromUtf8("longAxisPlotCheckbox"))
        self.PlotsverticalLayout.addWidget(self.longAxisPlotCheckbox)
        self.shortAxisPlotCheckbox = QtGui.QCheckBox(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.shortAxisPlotCheckbox.setFont(font)
        self.shortAxisPlotCheckbox.setObjectName(_fromUtf8("shortAxisPlotCheckbox"))
        self.PlotsverticalLayout.addWidget(self.shortAxisPlotCheckbox)
        self.diagonalAxisPlotCheckbox = QtGui.QCheckBox(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.diagonalAxisPlotCheckbox.setFont(font)
        self.diagonalAxisPlotCheckbox.setObjectName(_fromUtf8("diagonalAxisPlotCheckbox"))
        self.PlotsverticalLayout.addWidget(self.diagonalAxisPlotCheckbox)
        self.surfacePlotCheckbox = QtGui.QCheckBox(self.layoutWidget_2)
        self.surfacePlotCheckbox.setObjectName(_fromUtf8("surfacePlotCheckbox"))
        self.PlotsverticalLayout.addWidget(self.surfacePlotCheckbox)
        self.uniformityBoxSizeCheckbox = QtGui.QCheckBox(self.layoutWidget_2)
        self.uniformityBoxSizeCheckbox.setObjectName(_fromUtf8("uniformityBoxSizeCheckbox"))
        self.PlotsverticalLayout.addWidget(self.uniformityBoxSizeCheckbox)
        spacerItem2 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.PlotsverticalLayout.addItem(spacerItem2)
        self.processPushButton = QtGui.QPushButton(self.layoutWidget_2)
        self.processPushButton.setObjectName(_fromUtf8("processPushButton"))
        self.PlotsverticalLayout.addWidget(self.processPushButton)
        self.layoutWidget = QtGui.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 364, 306))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setMargin(0)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.inputFileLabel = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.inputFileLabel.setFont(font)
        self.inputFileLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.inputFileLabel.setObjectName(_fromUtf8("inputFileLabel"))
        self.horizontalLayout_5.addWidget(self.inputFileLabel)
        self.inputFileLineEdit = QtGui.QLineEdit(self.layoutWidget)
        self.inputFileLineEdit.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.inputFileLineEdit.setFont(font)
        self.inputFileLineEdit.setObjectName(_fromUtf8("inputFileLineEdit"))
        self.horizontalLayout_5.addWidget(self.inputFileLineEdit)
        self.inputFileBrowsePushButton = QtGui.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.inputFileBrowsePushButton.setFont(font)
        self.inputFileBrowsePushButton.setAutoDefault(False)
        self.inputFileBrowsePushButton.setDefault(True)
        self.inputFileBrowsePushButton.setObjectName(_fromUtf8("inputFileBrowsePushButton"))
        self.horizontalLayout_5.addWidget(self.inputFileBrowsePushButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        spacerItem3 = QtGui.QSpacerItem(35, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem3)
        self.scanNameLabel = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.scanNameLabel.setFont(font)
        self.scanNameLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.scanNameLabel.setObjectName(_fromUtf8("scanNameLabel"))
        self.horizontalLayout_4.addWidget(self.scanNameLabel)
        self.scanNameLineEdit = QtGui.QLineEdit(self.layoutWidget)
        self.scanNameLineEdit.setMinimumSize(QtCore.QSize(263, 0))
        self.scanNameLineEdit.setMaximumSize(QtCore.QSize(305, 16777215))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.scanNameLineEdit.setFont(font)
        self.scanNameLineEdit.setObjectName(_fromUtf8("scanNameLineEdit"))
        self.horizontalLayout_4.addWidget(self.scanNameLineEdit)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        spacerItem4 = QtGui.QSpacerItem(55, 20, QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem4)
        self.apertureLabel = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.apertureLabel.setFont(font)
        self.apertureLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.apertureLabel.setObjectName(_fromUtf8("apertureLabel"))
        self.horizontalLayout_7.addWidget(self.apertureLabel)
        self.apertureComboBox = QtGui.QComboBox(self.layoutWidget)
        self.apertureComboBox.setObjectName(_fromUtf8("apertureComboBox"))
        self.apertureComboBox.addItem(_fromUtf8(""))
        self.apertureComboBox.addItem(_fromUtf8(""))
        self.apertureComboBox.addItem(_fromUtf8(""))
        self.apertureComboBox.addItem(_fromUtf8(""))
        self.horizontalLayout_7.addWidget(self.apertureComboBox)
        spacerItem5 = QtGui.QSpacerItem(60, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem5)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.widthProfileLabel = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.widthProfileLabel.setFont(font)
        self.widthProfileLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.widthProfileLabel.setObjectName(_fromUtf8("widthProfileLabel"))
        self.horizontalLayout_8.addWidget(self.widthProfileLabel)
        self.widthProfileSpinBox = QtGui.QSpinBox(self.layoutWidget)
        self.widthProfileSpinBox.setMaximumSize(QtCore.QSize(50, 16777215))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.widthProfileSpinBox.setFont(font)
        self.widthProfileSpinBox.setMinimum(25)
        self.widthProfileSpinBox.setMaximum(500)
        self.widthProfileSpinBox.setProperty("value", 100)
        self.widthProfileSpinBox.setObjectName(_fromUtf8("widthProfileSpinBox"))
        self.horizontalLayout_8.addWidget(self.widthProfileSpinBox)
        self.mmLabel = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.mmLabel.setFont(font)
        self.mmLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.mmLabel.setObjectName(_fromUtf8("mmLabel"))
        self.horizontalLayout_8.addWidget(self.mmLabel)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.heightProfileLabel = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.heightProfileLabel.setFont(font)
        self.heightProfileLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.heightProfileLabel.setObjectName(_fromUtf8("heightProfileLabel"))
        self.horizontalLayout_9.addWidget(self.heightProfileLabel)
        self.heightProfileSpinBox = QtGui.QSpinBox(self.layoutWidget)
        self.heightProfileSpinBox.setMaximumSize(QtCore.QSize(50, 16777215))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.heightProfileSpinBox.setFont(font)
        self.heightProfileSpinBox.setMinimum(25)
        self.heightProfileSpinBox.setMaximum(140)
        self.heightProfileSpinBox.setProperty("value", 100)
        self.heightProfileSpinBox.setObjectName(_fromUtf8("heightProfileSpinBox"))
        self.horizontalLayout_9.addWidget(self.heightProfileSpinBox)
        self.mmLabel1 = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.mmLabel1.setFont(font)
        self.mmLabel1.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.mmLabel1.setObjectName(_fromUtf8("mmLabel1"))
        self.horizontalLayout_9.addWidget(self.mmLabel1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_9)
        spacerItem6 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem6)
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.label = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout_3.addWidget(self.label)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.pixelPitchLabel_2 = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.pixelPitchLabel_2.setFont(font)
        self.pixelPitchLabel_2.setObjectName(_fromUtf8("pixelPitchLabel_2"))
        self.horizontalLayout_3.addWidget(self.pixelPitchLabel_2)
        self.powerBoundarySpinBox = QtGui.QSpinBox(self.layoutWidget)
        self.powerBoundarySpinBox.setMinimum(1)
        self.powerBoundarySpinBox.setProperty("value", 80)
        self.powerBoundarySpinBox.setObjectName(_fromUtf8("powerBoundarySpinBox"))
        self.horizontalLayout_3.addWidget(self.powerBoundarySpinBox)
        self.apertureLabel_6 = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.apertureLabel_6.setFont(font)
        self.apertureLabel_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.apertureLabel_6.setObjectName(_fromUtf8("apertureLabel_6"))
        self.horizontalLayout_3.addWidget(self.apertureLabel_6)
        spacerItem7 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem7)
        self.pixelPitchLabel = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.pixelPitchLabel.setFont(font)
        self.pixelPitchLabel.setObjectName(_fromUtf8("pixelPitchLabel"))
        self.horizontalLayout_3.addWidget(self.pixelPitchLabel)
        self.pixelPitchDoubleSpinBox = QtGui.QDoubleSpinBox(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.pixelPitchDoubleSpinBox.setFont(font)
        self.pixelPitchDoubleSpinBox.setDecimals(1)
        self.pixelPitchDoubleSpinBox.setMinimum(0.1)
        self.pixelPitchDoubleSpinBox.setMaximum(5.0)
        self.pixelPitchDoubleSpinBox.setSingleStep(0.1)
        self.pixelPitchDoubleSpinBox.setProperty("value", 0.5)
        self.pixelPitchDoubleSpinBox.setObjectName(_fromUtf8("pixelPitchDoubleSpinBox"))
        self.horizontalLayout_3.addWidget(self.pixelPitchDoubleSpinBox)
        self.apertureLabel_5 = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.apertureLabel_5.setFont(font)
        self.apertureLabel_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.apertureLabel_5.setObjectName(_fromUtf8("apertureLabel_5"))
        self.horizontalLayout_3.addWidget(self.apertureLabel_5)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.autoSaveFigsCheckbox = QtGui.QCheckBox(self.layoutWidget)
        self.autoSaveFigsCheckbox.setChecked(True)
        self.autoSaveFigsCheckbox.setObjectName(_fromUtf8("autoSaveFigsCheckbox"))
        self.horizontalLayout_6.addWidget(self.autoSaveFigsCheckbox)
        spacerItem8 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem8)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_10 = QtGui.QHBoxLayout()
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        self.exportInterpolatedCheckbox = QtGui.QCheckBox(self.layoutWidget)
        self.exportInterpolatedCheckbox.setObjectName(_fromUtf8("exportInterpolatedCheckbox"))
        self.horizontalLayout_10.addWidget(self.exportInterpolatedCheckbox)
        spacerItem9 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem9)
        self.verticalLayout_3.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_11 = QtGui.QHBoxLayout()
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        self.label_3 = QtGui.QLabel(self.layoutWidget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_11.addWidget(self.label_3)
        spacerItem10 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem10)
        self.verticalLayout_3.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.colormapComboBox = QtGui.QComboBox(self.layoutWidget)
        self.colormapComboBox.setObjectName(_fromUtf8("colormapComboBox"))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.colormapComboBox.addItem(_fromUtf8(""))
        self.horizontalLayout_2.addWidget(self.colormapComboBox)
        self.colormapReverseCheckbox = QtGui.QCheckBox(self.layoutWidget)
        self.colormapReverseCheckbox.setChecked(False)
        self.colormapReverseCheckbox.setObjectName(_fromUtf8("colormapReverseCheckbox"))
        self.horizontalLayout_2.addWidget(self.colormapReverseCheckbox)
        spacerItem11 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem11)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.addLayout(self.verticalLayout_3)
        mainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(mainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        mainWindow.setWindowTitle(_translate("mainWindow", "2D Scan Post Processor v0.2", None))
        self.desiredPlotslabel.setText(_translate("mainWindow", "Desired Plots", None))
        self.heatMapCheckbox.setToolTip(_translate("mainWindow", "<html><head/><body><p><img src=\":/images/heat_map.png\" width=\"400\" height=\"300\"/></p></body></html>", None))
        self.heatMapCheckbox.setText(_translate("mainWindow", "Heat Map", None))
        self.contourPlotCheckbox.setToolTip(_translate("mainWindow", "<html><head/><body><p><img height=\"300\" width=\"400\" src=\":/images/contour_plot.png\"/></p></body></html>", None))
        self.contourPlotCheckbox.setText(_translate("mainWindow", "Contour Plot", None))
        self.contourLinesLabel.setToolTip(_translate("mainWindow", "<html><head/><body><p>Number of lines to draw in the topographical plot.</p><p>If 5, plot will have lines at 0, 20, 40, 60 and 80%.</p></body></html>", None))
        self.contourLinesLabel.setText(_translate("mainWindow", "Contour Lines", None))
        self.uniformityPlotCheckbox.setToolTip(_translate("mainWindow", "<html><head/><body><p><img height=\"300\" width=\"400\" src=\":/images/uniformity_plot.png\"/></p></body></html>", None))
        self.uniformityPlotCheckbox.setText(_translate("mainWindow", "Uniformity Plot", None))
        self.longAxisPlotCheckbox.setToolTip(_translate("mainWindow", "<html><head/><body><p><img height=\"300\" width=\"400\" src=\":/images/long_axis_plot.png\"/></p></body></html>", None))
        self.longAxisPlotCheckbox.setText(_translate("mainWindow", "Long Axis Plot", None))
        self.shortAxisPlotCheckbox.setToolTip(_translate("mainWindow", "<html><head/><body><p><img height=\"300\" width=\"400\" src=\":/images/short_axis_plot.png\"/></p></body></html>", None))
        self.shortAxisPlotCheckbox.setText(_translate("mainWindow", "Short Axis Plot", None))
        self.diagonalAxisPlotCheckbox.setToolTip(_translate("mainWindow", "<html><head/><body><p><img height=\"300\" width=\"400\" src=\":/images/diagonal_axis_plot.png\"/></p></body></html>", None))
        self.diagonalAxisPlotCheckbox.setText(_translate("mainWindow", "Diagonal Axis Plot", None))
        self.surfacePlotCheckbox.setToolTip(_translate("mainWindow", "<html><head/><body><p><img height=\"300\" width=\"400\" src=\":/images/surface_plot.png\"/></p></body></html>", None))
        self.surfacePlotCheckbox.setText(_translate("mainWindow", "3D Surface Plot", None))
        self.uniformityBoxSizeCheckbox.setToolTip(_translate("mainWindow", "<html><head/><body><p>Plot for confirming uniformity sample points (only for Doug)</p></body></html>", None))
        self.uniformityBoxSizeCheckbox.setText(_translate("mainWindow", "Uniformity vs. Box Size Ratio", None))
        self.processPushButton.setText(_translate("mainWindow", "Process", None))
        self.inputFileLabel.setText(_translate("mainWindow", "Data File to Process", None))
        self.inputFileLineEdit.setPlaceholderText(_translate("mainWindow", "ex: C:\\users\\ogi\\desktop\\data.csv", None))
        self.inputFileBrowsePushButton.setText(_translate("mainWindow", "Browse", None))
        self.scanNameLabel.setText(_translate("mainWindow", "Scan Name", None))
        self.scanNameLineEdit.setPlaceholderText(_translate("mainWindow", "ex:FJ800-65mm reflector-10mm offset", None))
        self.apertureLabel.setToolTip(_translate("mainWindow", "<html><head/><body><p>The diameter of the aperture of the integration sphere.  By default is 12.5mm, we have 1mm and 5mm apertures as well.</p></body></html>", None))
        self.apertureLabel.setText(_translate("mainWindow", "Aperture Diameter", None))
        self.apertureComboBox.setItemText(0, _translate("mainWindow", "12.5mm", None))
        self.apertureComboBox.setItemText(1, _translate("mainWindow", "10.0mm", None))
        self.apertureComboBox.setItemText(2, _translate("mainWindow", "5.0mm", None))
        self.apertureComboBox.setItemText(3, _translate("mainWindow", "1.0mm", None))
        self.widthProfileLabel.setToolTip(_translate("mainWindow", "<html><head/><body><p>Expected Light Width Profile.  Example: FJ100-75 is 75mm</p></body></html>", None))
        self.widthProfileLabel.setText(_translate("mainWindow", "Lamp Width", None))
        self.mmLabel.setText(_translate("mainWindow", "(mm)", None))
        self.heightProfileLabel.setToolTip(_translate("mainWindow", "<html><head/><body><p>Expected light profile.  Example: FJ800 has 100mm x 100mm</p></body></html>", None))
        self.heightProfileLabel.setText(_translate("mainWindow", "Lamp Height", None))
        self.mmLabel1.setText(_translate("mainWindow", "(mm)", None))
        self.label.setText(_translate("mainWindow", "Extra Options", None))
        self.pixelPitchLabel_2.setToolTip(_translate("mainWindow", "<html><head/><body><p>Used in Uniformity Plot, draws boundary at specified percentage.</p></body></html>", None))
        self.pixelPitchLabel_2.setText(_translate("mainWindow", "Power Boundary", None))
        self.apertureLabel_6.setText(_translate("mainWindow", "(%)  ", None))
        self.pixelPitchLabel.setToolTip(_translate("mainWindow", "<html><head/><body><p>This determines the spacing between interpolated points. Lower values result in smoother plots, but longer computation time.</p><p>Typical values:</p><p>Default: 0.5mm</p><p>Smooth: 0.2mm</p><p>Coarse: 1.0mm</p></body></html>", None))
        self.pixelPitchLabel.setText(_translate("mainWindow", "Pixel Pitch", None))
        self.apertureLabel_5.setText(_translate("mainWindow", "(mm)", None))
        self.autoSaveFigsCheckbox.setToolTip(_translate("mainWindow", "<html><head/><body><p>Automatically save plots as PNG files in same directory as data file.</p></body></html>", None))
        self.autoSaveFigsCheckbox.setText(_translate("mainWindow", "Auto-Save Figures", None))
        self.exportInterpolatedCheckbox.setToolTip(_translate("mainWindow", "<html><head/><body><p>Generates CSV files from linear scan passes that can be used in further post-processing.</p></body></html>", None))
        self.exportInterpolatedCheckbox.setText(_translate("mainWindow", "Export Interpolated Data", None))
        self.label_3.setToolTip(_translate("mainWindow", "<html><head/><body><p><img src=\":/images/colorMaps.png\"/></p></body></html>", None))
        self.label_3.setText(_translate("mainWindow", "Colormap", None))
        self.colormapComboBox.setItemText(0, _translate("mainWindow", "Cube Helix", None))
        self.colormapComboBox.setItemText(1, _translate("mainWindow", "Red", None))
        self.colormapComboBox.setItemText(2, _translate("mainWindow", "Blue", None))
        self.colormapComboBox.setItemText(3, _translate("mainWindow", "Green", None))
        self.colormapComboBox.setItemText(4, _translate("mainWindow", "Purple", None))
        self.colormapComboBox.setItemText(5, _translate("mainWindow", "Orange", None))
        self.colormapComboBox.setItemText(6, _translate("mainWindow", "Blue-Green", None))
        self.colormapComboBox.setItemText(7, _translate("mainWindow", "Blue-Purple", None))
        self.colormapComboBox.setItemText(8, _translate("mainWindow", "Green-Blue", None))
        self.colormapComboBox.setItemText(9, _translate("mainWindow", "Orange-Red", None))
        self.colormapComboBox.setItemText(10, _translate("mainWindow", "Purple-Blue", None))
        self.colormapComboBox.setItemText(11, _translate("mainWindow", "Purple-Blue-Green", None))
        self.colormapComboBox.setItemText(12, _translate("mainWindow", "Purple-Red", None))
        self.colormapComboBox.setItemText(13, _translate("mainWindow", "Red-Purple", None))
        self.colormapComboBox.setItemText(14, _translate("mainWindow", "Yellow-Green", None))
        self.colormapComboBox.setItemText(15, _translate("mainWindow", "Yellow-Green-Blue", None))
        self.colormapComboBox.setItemText(16, _translate("mainWindow", "Yellow-Orange-Brown", None))
        self.colormapComboBox.setItemText(17, _translate("mainWindow", "Yellow-Orange-Red", None))
        self.colormapComboBox.setItemText(18, _translate("mainWindow", "Flag", None))
        self.colormapComboBox.setItemText(19, _translate("mainWindow", "Jet", None))
        self.colormapReverseCheckbox.setToolTip(_translate("mainWindow", "<html><head/><body><p>Reverses the colormap ex: blue=hot &amp; red=cold</p><p><br/></p></body></html>", None))
        self.colormapReverseCheckbox.setText(_translate("mainWindow", "Reverse", None))

from ui_files import images_rc
