__appname__ = "2D Scan Post Processor"
__module__ = "main"

#####################################
# PS$: python <path>\pyuic.py mainWindow_revXX.ui -o mainWindow_revXX.py
# in the mainWindow_rev*.py file change:
# import images_rc
# to
# from ui_files import images_rc
#####################################


from PyQt4.QtGui import QMainWindow
from PyQt4.QtGui import QIcon
from PyQt4.QtGui import QFileDialog
from PyQt4.QtGui import QMessageBox
from PyQt4.QtGui import QApplication
from PyQt4.QtGui import QErrorMessage
from PyQt4.QtGui import QCloseEvent
from PyQt4.QtCore import QSettings
from PyQt4.QtCore import QCoreApplication

import postProcessor
import os
import sys
#import subprocess
import os.path
from ui_files import mainWindow_rev02 as pyMainWindow

appDataPath = os.environ["APPDATA"] + "\\Phoseon\\PostProcessor\\"

if not os.path.exists(appDataPath):
    try:
        os.makedirs(appDataPath)
    except Exception:
        appDataPath = os.getcwd()

class MainDialog(QMainWindow, pyMainWindow.Ui_mainWindow):

    config = appDataPath + "config.ini"

    def __init__(self, parent=None):
        super(MainDialog, self).__init__(parent)
        #Lines to generate the taskbar icon
        self.setWindowTitle("Post Processor")
        self.setWindowIcon(QIcon('.ui_files\images\phologo.png'))
        self.show()
        self.setupUi(self)
        self.settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "Post Processor", "Post Processor")
        self.load_initial_settings()

        #push button connectors
        self.inputFileBrowsePushButton.clicked.connect(self.browse_button_clicked)
        self.processPushButton.clicked.connect(self.process_button_clicked)

    def load_initial_settings(self):
        """
        Loads the previously stores values on the UI
        """
        #TODO: save previous entries and restore upon app opening
        pass

    def browse_button_clicked(self):
        """
        opens up a file browse button
        """
        dataFile = QFileDialog.getOpenFileNameAndFilter(parent=None,
                                                        caption="Import Data file",
                                                        directory=".",
                                                        filter="2-Axis Stage CSV Output (*.csv);;Simulated Data (*.txt)")
        if dataFile[0]:
            try:
                self.inputFileLineEdit.setText(dataFile[0])
            except Exception as e:
                QMessageBox.critical(self, __appname__, "Error importing file, error is \r\n" + str(e))
                return

    def process_button_clicked(self):
        """ Process the csv file with the options selected """

        args = {}
        args['fileName'] = str(self.inputFileLineEdit.text())
        args['gridResolution'] = 10
        args['contourResolution'] = self.contourLinesSpinBox.value()
        args['scanName'] = str(self.scanNameLineEdit.text())
        args['powerBoundaryPercentage'] = self.powerBoundarySpinBox.value() / 100.0
        args['pixelPitch'] = self.pixelPitchDoubleSpinBox.value()
        args['targetWidth'] = self.widthProfileSpinBox.value()
        args['targetHeight'] = self.heightProfileSpinBox.value()
        args['contourPlot'] = self.contourPlotCheckbox.isChecked()
        args['heatMap'] = self.heatMapCheckbox.isChecked()
        args['longAxisPlot'] = self.longAxisPlotCheckbox.isChecked()
        args['shortAxisPlot'] = self.shortAxisPlotCheckbox.isChecked()
        args['diagonalAxisPlot'] = self.diagonalAxisPlotCheckbox.isChecked()
        args['surfacePlot'] = self.surfacePlotCheckbox.isChecked()
        args['uniformityPlot'] = self.uniformityPlotCheckbox.isChecked()
        args['uniformityVsBoxSizeRatioPlot'] = self.uniformityBoxSizeCheckbox.isChecked()
        args['aperture'] = float(str(self.apertureComboBox.currentText()).split('mm')[0])
        args['autoSaveFigures'] = self.autoSaveFigsCheckbox.isChecked()
        args['csvExport'] = self.exportInterpolatedCheckbox.isChecked()
        args['colorMapReverse'] = self.colormapReverseCheckbox.isChecked()
        args['colorMap'] = str(self.colormapComboBox.currentText()).lower()
        

        if str(self.inputFileLineEdit.text()).split('.')[-1] == 'txt':
            args['simulatedData'] = True
        else:
            args['simulatedData'] = False

        #replace spaces with understores to handle colormaps
        #args['colorMap'] = args['colorMap'].replace(" ", "_")

        #if self.colormapReverseCheckbox.isChecked():
        #    args['colorMap'] += str('_r')
        #add this plot for Garth at a later time...
        args['heatMapAndUniformityPlot'] = False
        fault = postProcessor.process(args)

        if fault == 'pixel_pitch_fault':
            QErrorMessage.showMessage(self, fault)
            QCloseEvent(self)



def main(args):
    QCoreApplication.setApplicationName("2D Scan Post Processor")
    QCoreApplication.setApplicationVersion("0.2")
    QCoreApplication.setOrganizationName("Phoseon Technology")
    QCoreApplication.setOrganizationDomain("phoseon.com")

    app = QApplication([])
    form = MainDialog()
    #form.show()
    #app.exec_()
    sys.exit(app.exec_())
if __name__ == "__main__":
    main(sys.argv[1:])

