import numpy as np
import scipy.stats
from skimage import io
from skimage.color import gray2rgb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path
matplotlib.use("Qt5Agg")
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import os
import csv
import cv2


# TODO: Make it so that non-square grids can be used for full and rectangular crop (e.g., 8 strata turned into 4x2 grid)
# TODO: Figure out where to input number of strata (maybe add a step 5), or keep it fixed for now

# Class used to aid in displaying the image with grid overlayed onto sampled pixels
class PixelMap:
    def __init__(self,image:np.ndarray):
        self.rows = len(image)
        self.cols = len(image[0])
        self.numPixels = self.rows * self.cols
        self.originalImage = image # grayscale

        if np.max(image) <= 1.0:
            self.originalImage *= 255
            self.originalImage = self.originalImage.astype(int)

    # Overlay a grid onto the original image and return centered around that grid
    def GetImageWithGridOverlay(self, pixelRow:int, pixelCol:int, newColor:tuple, numSurroundingPixels:int, style:int) -> np.ndarray: 
        # Center
        displayImage = gray2rgb(self.originalImage)

        if style == 0:
            displayImage[pixelRow][pixelCol] = newColor

        minValue = 1 if style != 2 else 2
        
        # above
        maxVal = 3 if pixelRow > 2 else pixelRow
        for i in range(minValue, maxVal):
            displayImage[pixelRow-i][pixelCol] = newColor

        # below
        maxVal = 3 if pixelRow < self.rows-3 else self.rows-pixelRow
        for i in range(minValue, maxVal):
            displayImage[pixelRow+i][pixelCol] = newColor

        # right
        maxVal = 3 if pixelCol < self.cols-3 else self.cols-pixelCol
        for i in range(minValue, maxVal):
            displayImage[pixelRow][pixelCol+i] = newColor

        # left
        maxVal = 3 if pixelCol > 2 else pixelCol
        for i in range(minValue, maxVal):
            displayImage[pixelRow][pixelCol-i] = newColor
        
        # pad image to ensure display proper
        displayImage = np.pad(displayImage, ((numSurroundingPixels+1, numSurroundingPixels+1), (numSurroundingPixels+1, numSurroundingPixels+1), (0,0)))

        # Crop the image to center around the grid with numSurroundingPixels around
        displayImage = displayImage[pixelRow:pixelRow+2*numSurroundingPixels,pixelCol:pixelCol+2*numSurroundingPixels,:]

        return displayImage

    def GetCroppedImage(self, leftBound, rightBound, topBound, bottomBound):
        return self.originalImage[topBound:bottomBound, leftBound:rightBound]
    
    def GetCroppedAndMaskedImage(self, leftBound, rightBound, topBound, bottomBound, polygonPoints):
        return self.originalImage[topBound:bottomBound, leftBound:rightBound]


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class SetupWidget(QtWidgets.QWidget):
    def __init__(self, parentTab):
        super(SetupWidget, self).__init__()

        self.parentTab = parentTab
        self.countAreaBounds = None

        # Step 1 widgets and layout
        self.step1Number = QtWidgets.QLabel("1")
        self.step1Number.setStyleSheet("border: 3px solid black; border-radius: 40px; font: bold 24px")
        self.selectImageText = QtWidgets.QLabel("Select Image:")
        self.imagePathBox = QtWidgets.QLineEdit("")
        self.browseImagePath = QtWidgets.QPushButton("Browse")

        step1layout = QtWidgets.QHBoxLayout()
        step1layout.addWidget(self.step1Number)
        step1layout.addWidget(self.selectImageText)
        step1layout.addWidget(self.imagePathBox, stretch=2)
        step1layout.addWidget(self.browseImagePath)

        self.step1Widget = QtWidgets.QWidget()
        self.step1Widget.setLayout(step1layout)

        # Step 2 widgets and layout
        self.step2Number = QtWidgets.QLabel("2")
        self.step2Number.setStyleSheet("border: 3px solid black; border-radius: 40px; font: bold 24px")
        self.selectCountAreaText = QtWidgets.QLabel("Select Count Area")
        self.selectFullImageButton = QtWidgets.QPushButton("Full Image")
        self.selectFullImageButton.setCheckable(True)
        self.selectRectCropButton = QtWidgets.QPushButton("Rectangular Crop")
        self.selectRectCropButton.setCheckable(True)
        self.selectCircCropButton = QtWidgets.QPushButton("Circular Crop")
        self.selectCircCropButton.setCheckable(True)
        self.selectAnnularCropButton = QtWidgets.QPushButton("Annular Crop")
        self.selectAnnularCropButton.setCheckable(True)
        
        self.selectFullImageButton.setDisabled(True)
        self.selectRectCropButton.setDisabled(True)
        self.selectCircCropButton.setDisabled(True)
        self.selectAnnularCropButton.setDisabled(True)

        step2layout = QtWidgets.QHBoxLayout()
        step2layout.addWidget(self.step2Number)
        step2layout.addWidget(self.selectCountAreaText,stretch=2)
        step2layout.addWidget(self.selectFullImageButton,stretch=2)
        step2layout.addWidget(self.selectRectCropButton,stretch=2)
        step2layout.addWidget(self.selectCircCropButton,stretch=2)
        step2layout.addWidget(self.selectAnnularCropButton,stretch=2)

        self.step2Widget = QtWidgets.QWidget()
        self.step2Widget.setLayout(step2layout)

        # Step 3 widgets and layout
        self.step3Number = QtWidgets.QLabel("3")
        self.step3Number.setStyleSheet("border: 3px solid black; border-radius: 40px; font: bold 24px")
        self.selectAllocationText = QtWidgets.QLabel("Select Allocation Method:")
        self.selectProportionalButton = QtWidgets.QPushButton("Proportional")
        self.selectProportionalButton.setCheckable(True)
        self.selectOptimalButton = QtWidgets.QPushButton("Optimal")
        self.selectOptimalButton.setCheckable(True)

        step3layout = QtWidgets.QHBoxLayout()
        step3layout.addWidget(self.step3Number)
        step3layout.addWidget(self.selectAllocationText,stretch=2)
        step3layout.addWidget(self.selectProportionalButton,stretch=2)
        step3layout.addWidget(self.selectOptimalButton,stretch=2)

        self.step3Widget = QtWidgets.QWidget()
        self.step3Widget.setLayout(step3layout)

        # Step 4 widgets and layout
        self.step4Number = QtWidgets.QLabel("4")
        self.step4Number.setStyleSheet("border: 3px solid black; border-radius: 40px; font: bold 24px")
        self.setCItext = QtWidgets.QLabel("Set CI:")
        self.setCIbox = QtWidgets.QLineEdit("")
        self.setMOEtext = QtWidgets.QLabel("Set MOE:")
        self.setMOEbox = QtWidgets.QLineEdit("")

        step4layout = QtWidgets.QHBoxLayout()
        step4layout.addWidget(self.step4Number)
        step4layout.addWidget(self.setCItext)
        step4layout.addWidget(self.setCIbox)
        step4layout.addWidget(self.setMOEtext)
        step4layout.addWidget(self.setMOEbox)

        self.step4Widget = QtWidgets.QWidget()
        self.step4Widget.setLayout(step4layout)

        # Begin measurement button
        self.beginMeasurementButton = QtWidgets.QPushButton("Begin Measurement")

        # Export results button
        self.exportPreviousResultsButton = QtWidgets.QPushButton("Export previous results to csv")
        
        # Previous results table
        self.previousResultsTable = QtWidgets.QTableWidget()
        self.previousResultsTable.setRowCount(1)
        self.previousResultsTable.setColumnCount(5)
        self.previousResultsTable.setItem(0,0, QtWidgets.QTableWidgetItem("Image Name"))
        self.previousResultsTable.setItem(0,1, QtWidgets.QTableWidgetItem("Allocation"))
        self.previousResultsTable.setItem(0,2, QtWidgets.QTableWidgetItem("Area Fraction"))
        self.previousResultsTable.setItem(0,3, QtWidgets.QTableWidgetItem("Confidence Interval"))
        self.previousResultsTable.setItem(0,4, QtWidgets.QTableWidgetItem("Margin of Error"))
        self.previousResultsTable.setStyleSheet("border: 1px solid black; gridline-color: gray")
        self.previousResultsTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.previousResultsTable.verticalHeader().setVisible(False)
        self.previousResultsTable.horizontalHeader().setVisible(False)
        font = QtGui.QFont("Arial", 12)
        font.setBold(True)
        for i in range(5):
            self.previousResultsTable.item(0, i).setBackground(QtGui.QColor(196,217,244))
            self.previousResultsTable.item(0, i).setFont(font)
            self.previousResultsTable.item(0, i).setTextAlignment(QtCore.Qt.AlignCenter)

        # Empty space separator
        empty = QtWidgets.QFrame()

        # Full widget layout
        fullWidgetLayout = QtWidgets.QVBoxLayout()
        fullWidgetLayout.addWidget(empty, stretch=1)
        fullWidgetLayout.addWidget(self.step1Widget, stretch=1)
        fullWidgetLayout.addWidget(empty, stretch=1)
        fullWidgetLayout.addWidget(self.step2Widget, stretch=1)
        fullWidgetLayout.addWidget(empty, stretch=1)
        fullWidgetLayout.addWidget(self.step3Widget, stretch=1)
        fullWidgetLayout.addWidget(empty, stretch=1)
        fullWidgetLayout.addWidget(self.step4Widget, stretch=1)
        fullWidgetLayout.addWidget(empty, stretch=1)
        fullWidgetLayout.addWidget(self.beginMeasurementButton, stretch=1)
        fullWidgetLayout.addWidget(empty, stretch=1)
        fullWidgetLayout.addWidget(self.previousResultsTable)
        fullWidgetLayout.addWidget(self.exportPreviousResultsButton)

        self.setLayout(fullWidgetLayout)

        # Connect triggers
        self.imagePathBox.textChanged.connect(self.CheckImagePath)
        self.browseImagePath.clicked.connect(self.BrowseForImage)
        self.selectFullImageButton.clicked.connect(self.SelectFullImage)
        self.selectRectCropButton.clicked.connect(self.RectangularCrop)
        self.selectCircCropButton.clicked.connect(self.CircularCrop)
        self.selectAnnularCropButton.clicked.connect(self.AnnularCrop)
        self.selectOptimalButton.clicked.connect(self.SelectOptimal)
        self.selectProportionalButton.clicked.connect(self.SelectProportional)
        self.beginMeasurementButton.clicked.connect(self.BeginMeasurement)
        self.setMOEbox.textChanged.connect(self.CheckMOEandCI)
        self.setCIbox.textChanged.connect(self.CheckMOEandCI)
        self.exportPreviousResultsButton.clicked.connect(self.WriteResultsToCsv)
    
    def SelectOptimal(self):
        self.selectProportionalButton.setChecked(False)
        self.selectOptimalButton.setChecked(True)
        self.step3Number.setStyleSheet("border: 3px solid black; background-color: lightgreen; font: bold 24px")
        
    def SelectProportional(self):
        self.selectProportionalButton.setChecked(True)
        self.selectOptimalButton.setChecked(False)
        self.step3Number.setStyleSheet("border: 3px solid black; background-color: lightgreen; font: bold 24px")

    def SelectFullImage(self):
        self.selectFullImageButton.setChecked(True)
        self.selectRectCropButton.setChecked(False)
        self.selectCircCropButton.setChecked(False)
        self.selectAnnularCropButton.setChecked(False)

        self.countAreaBounds = None
        self.step2Number.setStyleSheet("border: 3px solid black; background-color: lightgreen; font: bold 24px")

    def RectangularCrop(self):
        self.selectFullImageButton.setChecked(False)
        self.selectRectCropButton.setChecked(True)
        self.selectCircCropButton.setChecked(False)
        self.selectAnnularCropButton.setChecked(False)

        img = cv2.imread(self.imagePathBox.text())

        # top left x, top left y, width, height
        self.countAreaBounds = cv2.selectROI("Select a ROI and then press ENTER button", img)
        cv2.destroyWindow('Select a ROI and then press ENTER button')

        if self.countAreaBounds[2] == 0 and self.countAreaBounds[3] == 0:
            self.step2Number.setStyleSheet("border: 3px solid black; background-color; font: bold 24px")
            self.selectRectCropButton.setChecked(False)
        else:
            self.step2Number.setStyleSheet("border: 3px solid black; background-color: lightgreen; font: bold 24px")

    def CircularCrop(self):
        self.selectFullImageButton.setChecked(False)
        self.selectRectCropButton.setChecked(False)
        self.selectCircCropButton.setChecked(True)
        self.selectAnnularCropButton.setChecked(False)

        coords = [None, None, None]
        img = cv2.imread(self.imagePathBox.text())

        # mouse callback function
        def draw_circle(event,x,y,flags,param):      
            if event == cv2.EVENT_LBUTTONUP:
                if param[0] is None:
                    param[0] = x
                    param[1] = y
                    cv2.circle(img,(x,y), 5, (0,0,255), -1)
                elif param[2] is None:
                    radius = int(np.sqrt((param[0]-x)**2 + (param[1]-y)**2))
                    if param[0] + radius > img.shape[1] or param[0] - radius < 0:
                        msg = QtWidgets.QMessageBox()
                        msg.setIcon(QtWidgets.QMessageBox.Critical)
                        msg.setText("Error")
                        msg.setInformativeText('Circle extends beyond image bounds.')
                        msg.setWindowTitle("Error")
                        msg.exec_()
                    elif param[1] + radius > img.shape[0] or param[1] - radius < 0:
                        msg = QtWidgets.QMessageBox()
                        msg.setIcon(QtWidgets.QMessageBox.Critical)
                        msg.setText("Error")
                        msg.setInformativeText('Circle extends beyond image bounds.')
                        msg.setWindowTitle("Error")
                        msg.exec_()
                    else:
                        param[2] = int(np.sqrt((param[0]-x)**2 + (param[1]-y)**2))
                        cv2.circle(img, (param[0], param[1]), param[2], (0, 0, 255), 2)

                        # find points along radius that divide circle into 16 strata
                        thetas = np.linspace(0,2*np.pi, 17)
                        for theta in thetas:
                            endPointX = int(param[2] * np.cos(theta))
                            endPointY = int(param[2] * np.sin(theta))
                            endPointX += param[0]
                            endPointY += param[1]
                            cv2.line(img, (param[0], param[1]), (endPointX, endPointY), (0, 0, 255), 2)
        
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',draw_circle, param=coords)

        while(1):
            cv2.imshow('image',img)
            k = cv2.waitKey(1) & 0xFF
            if k == 10 or k == 13 or k == ord("q"):
                break
        cv2.destroyAllWindows()

        if coords[0] is not None and coords[2] is not None:
            self.countAreaBounds = coords
            self.step2Number.setStyleSheet("border: 3px solid black; background-color: lightgreen; font: bold 24px")
        else:
            self.step2Number.setStyleSheet("border: 3px solid black; font: bold 24px")
            self.selectCircCropButton.setChecked(False)

    def AnnularCrop(self):
        self.selectFullImageButton.setChecked(False)
        self.selectRectCropButton.setChecked(False)
        self.selectCircCropButton.setChecked(False)
        self.selectAnnularCropButton.setChecked(True)

        coords = [None, None, None, None]
        img = cv2.imread(self.imagePathBox.text())

        # mouse callback function
        def draw_circle(event,x,y,flags,param):      
            if event == cv2.EVENT_LBUTTONUP:
                if param[0] is None:
                    param[0] = x
                    param[1] = y
                    cv2.circle(img,(x,y), 5, (0,0,255), -1)
                elif param[2] is None:
                    radius = int(np.sqrt((param[0]-x)**2 + (param[1]-y)**2))
                    if param[0] + radius > img.shape[1] or param[0] - radius < 0:
                        msg = QtWidgets.QMessageBox()
                        msg.setIcon(QtWidgets.QMessageBox.Critical)
                        msg.setText("Error")
                        msg.setInformativeText('Circle extends beyond image bounds.')
                        msg.setWindowTitle("Error")
                        msg.exec_()
                    elif param[1] + radius > img.shape[0] or param[1] - radius < 0:
                        msg = QtWidgets.QMessageBox()
                        msg.setIcon(QtWidgets.QMessageBox.Critical)
                        msg.setText("Error")
                        msg.setInformativeText('Circle extends beyond image bounds.')
                        msg.setWindowTitle("Error")
                        msg.exec_()
                    else:
                        param[2] = int(np.sqrt((param[0]-x)**2 + (param[1]-y)**2))
                        cv2.circle(img, (param[0], param[1]), param[2], (0, 0, 255), 2)
                elif param[3] is None:
                    radius = int(np.sqrt((param[0]-x)**2 + (param[1]-y)**2))
                    if param[0] + radius > img.shape[1] or param[0] - radius < 0:
                        msg = QtWidgets.QMessageBox()
                        msg.setIcon(QtWidgets.QMessageBox.Critical)
                        msg.setText("Error")
                        msg.setInformativeText('Circle extends beyond image bounds.')
                        msg.setWindowTitle("Error")
                        msg.exec_()
                    elif param[1] + radius > img.shape[0] or param[1] - radius < 0:
                        msg = QtWidgets.QMessageBox()
                        msg.setIcon(QtWidgets.QMessageBox.Critical)
                        msg.setText("Error")
                        msg.setInformativeText('Circle extends beyond image bounds.')
                        msg.setWindowTitle("Error")
                        msg.exec_()
                    else:
                        param[3] = int(np.sqrt((param[0]-x)**2 + (param[1]-y)**2))
                        cv2.circle(img, (param[0], param[1]), param[3], (0, 0, 255), 2)

                        # find points along radius that divide circle into 16 strata
                        thetas = np.linspace(0,2*np.pi, 17)
                        for theta in thetas:
                            startPointX = int(param[2] * np.cos(theta))
                            startPointY = int(param[2] * np.sin(theta))
                            startPointX += param[0]
                            startPointY += param[1]
                            endPointX = int(param[3] * np.cos(theta))
                            endPointY = int(param[3] * np.sin(theta))
                            endPointX += param[0]
                            endPointY += param[1]
                            cv2.line(img, (startPointX, startPointY), (endPointX, endPointY), (0, 0, 255), 2)
        
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',draw_circle, param=coords)

        while(1):
            cv2.imshow('image',img)
            k = cv2.waitKey(1) & 0xFF
            if k == 10 or k == 13 or k == ord("q"):
                break
        cv2.destroyAllWindows()

        if coords[0] is not None and coords[2] is not None and coords[3] is not None:
            self.step2Number.setStyleSheet("border: 3px solid black; background-color: lightgreen; font: bold 24px")
            self.countAreaBounds = coords
        else:
            self.step2Number.setStyleSheet("border: 3px solid black; font: bold 24px")
            self.selectAnnularCropButton.setChecked(False)

    def BrowseForImage(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self,'Select Image File','./')
        if fileName is not None:
            self.imagePathBox.setText(fileName[0])

    def BeginMeasurement(self):
        c1 = self.step1Number.palette().button().color().name()
        c2 = self.step2Number.palette().button().color().name()
        c3 = self.step3Number.palette().button().color().name()
        c4 = self.step4Number.palette().button().color().name()

        if c1 == "#90ee90" and c2 == "#90ee90" and c3 == "#90ee90" and c4 == "#90ee90":
            self.parentTab.MoveToInitialGuessWidget()
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Not all input steps have been completed.')
            msg.setWindowTitle("Error")
            msg.exec_()

    def CheckImagePath(self):
        try:
            io.imread(self.imagePathBox.text(), as_gray=True)
            self.step1Number.setStyleSheet("border: 3px solid black; background-color: lightgreen; font: bold 24px")
            self.selectFullImageButton.setEnabled(True)
            self.selectRectCropButton.setEnabled(True)
            self.selectCircCropButton.setEnabled(True)
            self.selectAnnularCropButton.setEnabled(True)
        except:
            self.step1Number.setStyleSheet("border: 3px solid black; background-color: yellow; font: bold 24px")
            self.selectFullImageButton.setChecked(False)
            self.selectRectCropButton.setChecked(False)
            self.selectCircCropButton.setChecked(False)
            self.selectAnnularCropButton.setChecked(False)

            self.selectFullImageButton.setDisabled(True)
            self.selectRectCropButton.setDisabled(True)
            self.selectCircCropButton.setDisabled(True)
            self.selectAnnularCropButton.setDisabled(True)
        
        self.step2Number.setStyleSheet("border: 3px solid black; font: bold 24px")

    def CheckMOEandCI(self):
        try:
            moe = self.setMOEbox.text()
            moe = moe.split("%")[0]
            moe = float(moe)

            ci = self.setCIbox.text()
            ci = ci.split("%")[0]
            ci = float(ci)

            self.step4Number.setStyleSheet("border: 3px solid black; background-color: lightgreen; font: bold 24px")
        except:
            self.step4Number.setStyleSheet("border: 3px solid black; font: bold 24px")

    def GetAllocationStrategy(self):
        if self.selectOptimalButton.isChecked():
            return "Optimal"
        elif self.selectProportionalButton.isChecked():
            return "Proportional"
        else:
            return None

    def GetConfidence(self):
        ci = self.setCIbox.text()
        ci = ci.split("%")[0]
        ci = float(ci)

        # put confidence into 0,1 interval
        if ci > 1:
            ci /= 100

        return ci

    def GetMOE(self):
        moe = self.setMOEbox.text()
        moe = moe.split("%")[0]
        moe = float(moe)

        if moe > 1:
            moe /= 100

        return moe

    def AddResultsToTable(self, p_st, lowerCL, upperCL):
        rowPosition = self.previousResultsTable.rowCount()
        self.previousResultsTable.insertRow(rowPosition)
        
        self.previousResultsTable.setItem(rowPosition,0, QtWidgets.QTableWidgetItem(f"{os.path.basename(self.imagePathBox.text())}"))
        self.previousResultsTable.setItem(rowPosition,1, QtWidgets.QTableWidgetItem(self.GetAllocationStrategy()))
        self.previousResultsTable.setItem(rowPosition,2, QtWidgets.QTableWidgetItem(f"{100*p_st:.2f}%"))
        self.previousResultsTable.setItem(rowPosition,3, QtWidgets.QTableWidgetItem(f"{int(self.GetConfidence()*100)}% CI: ({100*lowerCL:.1f}%, {100*upperCL:.1f}%)"))
        self.previousResultsTable.setItem(rowPosition,4, QtWidgets.QTableWidgetItem(f"{100*(upperCL-lowerCL)/2:.2f}%"))

    def Clear(self):
        # Reset step 1 text
        self.imagePathBox.setText("")

        # Reset number highlights
        self.step1Number.setStyleSheet("border: 3px solid black; border-radius: 40px; font: bold 24px")
        self.step2Number.setStyleSheet("border: 3px solid black; border-radius: 40px; font: bold 24px")
        self.step3Number.setStyleSheet("border: 3px solid black; border-radius: 40px; font: bold 24px")
        self.step4Number.setStyleSheet("border: 3px solid black; border-radius: 40px; font: bold 24px")
        

        # Reset step 2 buttons
        self.selectFullImageButton.setChecked(False)
        self.selectRectCropButton.setChecked(False)
        self.selectCircCropButton.setChecked(False)
        self.selectAnnularCropButton.setChecked(False)

        self.selectFullImageButton.setDisabled(True)
        self.selectRectCropButton.setDisabled(True)
        self.selectCircCropButton.setDisabled(True)
        self.selectAnnularCropButton.setDisabled(True)

        self.countAreaBounds = None

        # Reset step 3 buttons
        self.selectOptimalButton.setChecked(False)
        self.selectProportionalButton.setChecked(False)

        # Reset step 4 text
        self.setMOEbox.setText("")
        self.setCIbox.setText("")

    def WriteResultsToCsv(self):
        if self.previousResultsTable.rowCount() == 1:
            return
        else:
            if os.path.exists("AreaFractionResults.csv"):
                startRow = 1
            else:
                startRow = 0
            with open("AreaFractionResults.csv", "a", newline="") as file:
                writer = csv.writer(file)
                for i in range(startRow, self.previousResultsTable.rowCount()):
                    rowToWrite = []
                    for j in range(5):
                        rowToWrite.append(self.previousResultsTable.item(i, j).text())
                    
                    writer.writerow(rowToWrite)


class InitialGuessWidget(QtWidgets.QWidget):
    
    def __init__(self, parentTab):
        super(InitialGuessWidget,self).__init__()

        self.imagePath = None
        self.numStrata_N = None
        self.strataIndex = 0
        self.initialGuesses = []

        self.parentTab = parentTab
        
        # 5, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 95
        self.fivePctButton = QtWidgets.QPushButton("5%")
        self.fivePctButton.clicked.connect(lambda: self.LogEstimate(0.05))
        self.twelvePctButton = QtWidgets.QPushButton("12.5%")
        self.twelvePctButton.clicked.connect(lambda: self.LogEstimate(0.125))
        self.twntyFivePctButton = QtWidgets.QPushButton("25%")
        self.twntyFivePctButton.clicked.connect(lambda: self.LogEstimate(0.25))
        self.thirtySevenPctButton = QtWidgets.QPushButton("37.5%")
        self.thirtySevenPctButton.clicked.connect(lambda: self.LogEstimate(0.375))
        self.fiftyPctButton = QtWidgets.QPushButton("50%")
        self.fiftyPctButton.clicked.connect(lambda: self.LogEstimate(0.50))
        self.sixtyTwoPctButton = QtWidgets.QPushButton("62.5%")
        self.sixtyTwoPctButton.clicked.connect(lambda: self.LogEstimate(0.625))
        self.seventyFivePctButton = QtWidgets.QPushButton("75%")
        self.seventyFivePctButton.clicked.connect(lambda: self.LogEstimate(0.75))
        self.eightySevenPctButton = QtWidgets.QPushButton("87.5%")
        self.eightySevenPctButton.clicked.connect(lambda: self.LogEstimate(0.875))
        self.ninetyFivePctButton = QtWidgets.QPushButton("95%")
        self.ninetyFivePctButton.clicked.connect(lambda: self.LogEstimate(0.95))

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.fivePctButton)
        hbox.addWidget(self.twelvePctButton)
        hbox.addWidget(self.twntyFivePctButton)
        hbox.addWidget(self.thirtySevenPctButton)
        hbox.addWidget(self.fiftyPctButton)
        hbox.addWidget(self.sixtyTwoPctButton)
        hbox.addWidget(self.seventyFivePctButton)
        hbox.addWidget(self.eightySevenPctButton)
        hbox.addWidget(self.ninetyFivePctButton)
        
        self.buttonRegion = QtWidgets.QWidget()
        self.buttonRegion.setLayout(hbox)

        vbox = QtWidgets.QVBoxLayout()

        self.sc = MplCanvas(self, width=7, height=7, dpi=100)
        vbox.addWidget(self.sc)
        vbox.addWidget(self.buttonRegion)
        self.setLayout(vbox)

    def ReadImage(self, imagePath, numStrata, countAreaType, countAreaBounds=None):
        self.imagePath = imagePath
        self.originalImage = io.imread(imagePath, as_gray=True)
        self.numStrata = numStrata
        self.countAreaType = countAreaType
        self.countAreaBounds = countAreaBounds

        self.originalImage = io.imread(imagePath, as_gray=True)

        self.myMap = PixelMap(self.originalImage)
        self.N = self.myMap.numPixels
        
        self.strataIndex = 0
        self.DisplayStrata()

    def LogEstimate(self, value):
        self.initialGuesses.append(value)

        self.strataIndex += 1

        if self.strataIndex == self.numStrata:
            self.parentTab.MoveToConstituentCountWidget()
            return

        self.DisplayStrata()

    def DisplayStrata(self):

        # left,right,top,bottom bounds are rectangular bounds
        # for full and rectangular crop, this is equal to displayed region
        # for circular and annular, this is a box around the quarter circle in which the stratum lies

        if self.countAreaType == "Full":
            numStrata_N = int(np.sqrt(self.numStrata))
            unraveledStrataIndex = np.unravel_index(self.strataIndex, (numStrata_N, numStrata_N))
            row = unraveledStrataIndex[0]
            col = unraveledStrataIndex[1]

            topBound = int(row*self.myMap.rows/numStrata_N)
            bottomBound = int((row+1)*self.myMap.rows/numStrata_N)
            leftBound = int(col*self.myMap.cols/numStrata_N)
            rightBound = int((col+1)*self.myMap.cols/numStrata_N)

            image = self.myMap.GetCroppedImage(leftBound, rightBound, topBound, bottomBound)   
        
        elif self.countAreaType == "Rectangular":
            numStrata_N = int(np.sqrt(self.numStrata))
            unraveledStrataIndex = np.unravel_index(self.strataIndex, (numStrata_N, numStrata_N))
            row = unraveledStrataIndex[0]
            col = unraveledStrataIndex[1]

            topBound = int(self.countAreaBounds[1] + (row / numStrata_N) * self.countAreaBounds[3])
            bottomBound = int(self.countAreaBounds[1] + ((row+1) / numStrata_N) * self.countAreaBounds[3])
            leftBound = int(self.countAreaBounds[0] + (col / numStrata_N) * self.countAreaBounds[2])
            rightBound = int(self.countAreaBounds[0] + ((col+1) / numStrata_N) * self.countAreaBounds[2])

            image = self.myMap.GetCroppedImage(leftBound, rightBound, topBound, bottomBound)   
        
        # FIXME: Write this
        elif self.countAreaType == "Circular":
            # Square box around circle
            topBound = self.countAreaBounds[1] - self.countAreaBounds[2] 
            bottomBound = self.countAreaBounds[1] + self.countAreaBounds[2]
            leftBound = self.countAreaBounds[0] - self.countAreaBounds[2]
            rightBound = self.countAreaBounds[0] + self.countAreaBounds[2]

            # Divide square into appropriate quarter
            strataFraction = self.strataIndex / self.numStrata
            if strataFraction < 0.25:
                leftBound = self.countAreaBounds[0]
                bottomBound = self.countAreaBounds[1]
            elif strataFraction < 0.5:
                rightBound = self.countAreaBounds[0]
                bottomBound = self.countAreaBounds[1]
            elif strataFraction < 0.75:
                rightBound = self.countAreaBounds[0]
                topBound = self.countAreaBounds[1]
            else:
                leftBound = self.countAreaBounds[0]
                topBound = self.countAreaBounds[1]
            
            # find points along radius that divide circle into strata
            thetas = np.linspace(0,2*np.pi, self.numStrata)

            # further subdivide arc 
            fineThetas = np.linspace(thetas[self.strataIndex], thetas[self.strataIndex+1], 10)

            # store polygon points
            polygon = [(self.countAreaBounds[0], self.countAreaBounds[1])]
            for theta in fineThetas:
                x = int(self.countAreaBounds[2] * np.cos(theta) + self.countAreaBounds[0])
                y = int(self.countAreaBounds[2] * np.sin(theta) + self.countAreaBounds[1])

                polygon.append((x,y)) 
            
            # TODO: Implement some algorithm to find if points are within polygon
              
        # FIXME: Write this
        elif self.countAreaType == "Annular":
            # Square box around circle
            topBound = self.countAreaBounds[1] - self.countAreaBounds[3] 
            bottomBound = self.countAreaBounds[1] + self.countAreaBounds[3]
            leftBound = self.countAreaBounds[0] - self.countAreaBounds[3]
            rightBound = self.countAreaBounds[0] + self.countAreaBounds[3]

            # Divide square into appropriate quarter
            strataFraction = self.strataIndex / self.numStrata_N
            if strataFraction < 0.25:
                leftBound = self.countAreaBounds[0]
                bottomBound = self.countAreaBounds[1]
            elif strataFraction < 0.5:
                rightBound = self.countAreaBounds[0]
                bottomBound = self.countAreaBounds[1]
            elif strataFraction < 0.75:
                rightBound = self.countAreaBounds[0]
                topBound = self.countAreaBounds[1]
            else:
                leftBound = self.countAreaBounds[0]
                topBound = self.countAreaBounds[1]


        self.sc.axes.cla()
        self.sc.axes.imshow(image, cmap="gray", vmin=0, vmax=255)
        self.sc.axes.set_yticks([])
        self.sc.axes.set_xticks([])
        self.sc.draw()


class ConstituentCountingWidget(QtWidgets.QWidget):

    def __init__(self, parentTab):
        super(ConstituentCountingWidget, self).__init__()

        self.parentTab = parentTab

        self.allocationStrategy = None

        self.displayToggle = 0

        self.sc = MplCanvas(self, width=7, height=7, dpi=100)

        vbox = QtWidgets.QVBoxLayout()

        hbox2 = QtWidgets.QHBoxLayout()

        self.lastEntryText = QtWidgets.QLabel("Last Entry: --")
        self.lastEntryText.setAlignment(QtCore.Qt.AlignCenter)
        self.lastEntryText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        self.indexProgressText = QtWidgets.QLabel(f"Sample: -/-, Strata: -/-")
        self.indexProgressText.setAlignment(QtCore.Qt.AlignCenter)
        self.indexProgressText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        hbox2.addWidget(self.lastEntryText)
        hbox2.addWidget(self.indexProgressText)
        
        self.zoomOutButton = QtWidgets.QPushButton("Zoom Out")
        self.zoomOutButton.clicked.connect(self.ZoomOut)
        self.zoomInButton = QtWidgets.QPushButton("Zoom In")
        self.zoomInButton.clicked.connect(self.ZoomIn)

        hbox2.addWidget(self.zoomOutButton)
        hbox2.addWidget(self.zoomInButton)

        vbox.addLayout(hbox2)
        vbox.addWidget(self.sc)
        hbox = QtWidgets.QHBoxLayout()

        leftText = QtWidgets.QLabel("Left Arrow Key For 0")
        leftText.setAlignment (QtCore.Qt.AlignCenter)
        leftText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        upText = QtWidgets.QLabel("Up Arrow Key For 0.5")
        upText.setAlignment (QtCore.Qt.AlignCenter)
        upText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        rightText = QtWidgets.QLabel("Right Arrow Key For 1")
        rightText.setAlignment (QtCore.Qt.AlignCenter)
        rightText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        downText = QtWidgets.QLabel("Down Arrow Key To Go Back")
        downText.setAlignment (QtCore.Qt.AlignCenter)
        downText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.addWidget(upText)
        vbox2.addWidget(downText)

        hbox.addWidget(leftText)
        hbox.addLayout(vbox2)
        hbox.addWidget(rightText)
        vbox.addLayout(hbox)

        self.setLayout(vbox)
        
        self.numSurroundingPixels = 50

        self.e_moe = 0.01
        self.d = 0.9

    def InitializeCounting(self, initialGuesses, imagePath, allocationStrategy, countAreaBounds, countAreaType, confidence, MOE):

        self.numStrata_N = int(np.sqrt(len(initialGuesses)))
        self.imageName = imagePath
        image = io.imread(imagePath, as_gray=True)
        self.myMap = PixelMap(image)
        self.N = self.myMap.numPixels
        self.N_h = int(self.N / self.numStrata_N**2)
        self.confidence = confidence

        # ############################################################################ #
        # Calculate the total number of samples needed to acheieve specified precision #
        # ############################################################################ #

        initialStrataProportion = np.sum(initialGuesses) * self.N_h / self.N
        
        if initialStrataProportion > 0.5 and MOE > 1-initialStrataProportion:
            MOE = ((1-initialStrataProportion)+MOE) / 2 # want to keep +- MOE as close as possible on the open side. so if p=0.01, with 5% MOE, the CI should be (0,0.06)
            print(f"MOE stretches beyond range of [0,1] based on initial guess, reducing to {initialStrataProportion:.2f}")
        elif initialStrataProportion < 0.5 and MOE > initialStrataProportion:
            MOE = (initialStrataProportion + MOE) / 2
            print(f"MOE stretches beyond range of [0,1] based on initial guess, reducing to {initialStrataProportion:.2f}")
        
        upperCL = 1.0
        lowerCL = 0.0
        withinTolerance = False
        currentIter = 0
        maxIters = 1000
        currentIter = 0
        n = self.numStrata_N ** 2
        iterExponent = 1
        d = 0.5 * 0.75**(iterExponent-1) + 1
        while not withinTolerance and currentIter < maxIters:
            n = int(np.ceil(n))

            lowerCL, upperCL = scipy.stats.binom.interval(confidence, n, initialStrataProportion)
            lowerCL /= n
            upperCL /= n

            if ((upperCL - lowerCL) / 2) > MOE: # Eq.15 not satisfied
                n *= d
            else: # Eq. 15 satisfied
                pctDiff = abs((((upperCL - lowerCL) / 2) - MOE) / MOE)
                if pctDiff > self.e_moe: # overestimating how many sample points needed
                    n /= d
                    iterExponent += 1
                    d = 2 * 0.75**(iterExponent-1) + 1
                else:
                    withinTolerance = True
            currentIter += 1
        
        if currentIter == maxIters:
            raise(RuntimeError("Max Iterations reached"))

        # ###################################################### #
        # Allocate the total number of samples across the strata #
        # ###################################################### #

        def OptimalAllocation(n):
            W_h = 1/self.numStrata_N**2
            
            optRatio = (W_h * (np.sum(np.sqrt(initialGuesses * (1-initialGuesses))))**2) / (np.sum(initialGuesses * (1-initialGuesses)))
            n *= optRatio

            strataVars = np.empty(self.numStrata_N**2)
            cnt = 0
            for i in range(self.numStrata_N):
                topBound = int(i*self.myMap.rows/self.numStrata_N)
                bottomBound = int((i+1)*self.myMap.rows/self.numStrata_N)
                for j in range(self.numStrata_N):
                    leftBound = int(j*self.myMap.cols/self.numStrata_N)
                    rightBound = int((j+1)*self.myMap.cols/self.numStrata_N)
                    strataVars[cnt] = np.var(self.myMap.originalImage[topBound:bottomBound,leftBound:rightBound])
                    cnt += 1

            n_h = np.empty(self.numStrata_N**2)

            n_h = np.ceil(n * self.N_h * strataVars / np.sum(self.N_h * strataVars)).astype(int)

            # Ensure that at least one point sampled per strata
            n_h = np.maximum(n_h, 1)

            return n_h

        def ProportionalAllocation(n):
            n_h = np.ones(self.numStrata_N**2) * (n/self.numStrata_N**2)

            n_h = np.ceil(n_h).astype(int)

            return n_h

        if allocationStrategy == "Optimal":
            n_h = OptimalAllocation(n)
        else:
            n_h = ProportionalAllocation(n)

        self.n_h = n_h

        # ########################## #
        # Get pixel sample locations #
        # ########################## #


        # TODO: Use count area type and bounds to determine the sampled region
        pixels = []
        cnt = 0
        for i in range(self.numStrata_N):
            topBound = int(i*self.myMap.rows/self.numStrata_N)
            bottomBound = int((i+1)*self.myMap.rows/self.numStrata_N)
            for j in range(self.numStrata_N):
                leftBound = int(j*self.myMap.cols/self.numStrata_N)
                rightBound = int((j+1)*self.myMap.cols/self.numStrata_N)

                random = np.random.choice(np.arange(0,((bottomBound-topBound) * (rightBound-leftBound))), n_h[cnt], replace=False)
                random = np.array(np.unravel_index(random, (bottomBound-topBound,rightBound-leftBound)))
                random[0,:] += topBound
                random[1,:] += leftBound
        
                pixels.append(list(zip(random[0,:], random[1,:])))

                cnt += 1
        
        self.samplePositions = pixels # First axis is strata axis, second axis is sample axis
        self.numGrids = np.sum(n_h)

        # ############# #
        # Begin display #
        # ############# #

        self.sampleIndex = 0
        self.strataIndex = 0
        self.gridIndex = 0 # Used to track flattend index, useful for writing to 1D poreData

        self.poreData = np.zeros((self.numGrids))
        self.indexProgressText.setText(f"Sample: {self.sampleIndex+1}/{self.n_h[self.strataIndex]}, Strata: {self.strataIndex+1}/{self.numStrata_N**2}")

        self.UpdateDisplay()

        self.setFocus(QtCore.Qt.NoFocusReason) # Needed or the keyboard will not work

    def TwoSidedCL_A(self, n, p, confidence):
        sum = 0
        A_U = 0
        while sum <= confidence + (1-confidence)/2: # The addition term is used because confidence includes AUC of upper and lower bound.
            A_U += 1
            sum = 0
            for i in range(A_U):
                sum += scipy.stats.binom.pmf(i, n, p)

        sum = 1.0
        A_L = 0
        while sum > confidence + (1-confidence)/2: # The addition term is used because confidence includes AUC of upper and lower bound.
            A_L += 1
            sum = 0
            for i in range(A_L, n+1):
                sum += scipy.stats.binom.pmf(i, n, p)
        
        return A_L-1, A_U-1

    def ZoomOut(self):
        if self.numSurroundingPixels < 300:
            self.numSurroundingPixels += 25
        
        self.UpdateDisplay()

        if self.numSurroundingPixels >= 300:
            self.zoomOutButton.setEnabled(False)
        else:
            self.zoomOutButton.setEnabled(True)

        if self.numSurroundingPixels <= 25:
            self.zoomInButton.setEnabled(False)
        else:
            self.zoomInButton.setEnabled(True)
        
        self.setFocus(QtCore.Qt.NoFocusReason) # Needed or the keyboard will not work
    
    def ZoomIn(self):
        if self.numSurroundingPixels > 25:
            self.numSurroundingPixels -= 25
        
        self.UpdateDisplay()

        if self.numSurroundingPixels == 25:
            self.zoomInButton.setEnabled(False)
        else:
            self.zoomInButton.setEnabled(True)

        if self.numSurroundingPixels == 300:
            self.zoomOutButton.setEnabled(False)
        else:
            self.zoomOutButton.setEnabled(True)
        
        self.setFocus(QtCore.Qt.NoFocusReason) # Needed or the keyboard will not work
        
    def keyPressEvent(self, event):
        if self.parentTab.stackedWidget.currentIndex() != 2:
            return
        
        if event.key() == QtCore.Qt.Key_Left:
            self.RecordDataPoint(0)
            self.lastEntryText.setText("Last Data Entry: 0")
        elif event.key() == QtCore.Qt.Key_Up:
            self.RecordDataPoint(0.5)
            self.lastEntryText.setText("Last Data Entry: 0.5")
        elif event.key() == QtCore.Qt.Key_Right:
            self.RecordDataPoint(1)
            self.lastEntryText.setText("Last Data Entry: 1")
        elif event.key() == QtCore.Qt.Key_Down:
            self.RecordDataPoint(-1)
            self.lastEntryText.setText("Last Data Entry: Back")
        elif event.key() == QtCore.Qt.Key_Plus:
            self.ZoomIn()
        elif event.key() == QtCore.Qt.Key_Minus:
            self.ZoomOut()
        elif event.key() == QtCore.Qt.Key_H:
            self.ToggleDisplay()
        else:
            pass
    
    def RecordDataPoint(self, value):
        # -1 is go back
        if value == -1 and self.sampleIndex > 0: # move back one sample
            self.sampleIndex -= 1
            self.gridIndex -= 1
        elif value == -1 and self.sampleIndex == 0 and self.strataIndex > 0: # move to end of last strata
            self.strataIndex -= 1
            self.sampleIndex = self.n_h[self.strataIndex] - 1
            self.gridIndex -= 1
        elif value == -1 and self.sampleIndex == 0 and self.strataIndex == 0: # at the beginning of samples, do nothing
            pass
        else: # record value
            self.poreData[self.gridIndex] = value
            self.gridIndex += 1

            # At last grid index, move to next pore index
            if self.sampleIndex == self.n_h[self.strataIndex] - 1:
                self.sampleIndex = 0
                self.strataIndex += 1           
            else:
                self.sampleIndex += 1

        if self.gridIndex >= self.numGrids:
            p_h = []

            for i, n in enumerate(self.n_h):
                if i == 0:
                    bounds = [0, n]
                else:
                    bounds = [np.cumsum(self.n_h[:i])[-1], np.cumsum(self.n_h[:i])[-1] + n]
                p_h.append(np.average(self.poreData[bounds[0]:bounds[1]]))

            p_st = np.sum(p_h) * self.N_h / self.N

            p_h = np.array(p_h)

            lowerCL, upperCL = scipy.stats.binom.interval(self.confidence, np.sum(self.n_h), p_st)
            lowerCL /= np.sum(self.n_h)
            upperCL /= np.sum(self.n_h)

            
            self.parentTab.MoveToSetupWidget(p_st, lowerCL, upperCL)

            return

        self.indexProgressText.setText(f"Sample: {self.sampleIndex+1}/{self.n_h[self.strataIndex]}, Strata: {self.strataIndex+1}/{self.numStrata_N**2}")

        self.UpdateDisplay()

    def UpdateDisplay(self):
        displayImage = self.myMap.GetImageWithGridOverlay(self.samplePositions[self.strataIndex][self.sampleIndex][0], self.samplePositions[self.strataIndex][self.sampleIndex][1], (50, 225, 248), self.numSurroundingPixels, self.displayToggle)

        self.sc.axes.cla()
        self.sc.axes.imshow(displayImage)
        self.sc.axes.set_yticks([])
        self.sc.axes.set_xticks([])
        self.sc.draw()

    def ToggleDisplay(self):
        
        if self.displayToggle != 2:
            self.displayToggle += 1
        else:
            self.displayToggle = 0
        
        self.UpdateDisplay()


class MyWindow(QMainWindow):

    def __init__(self):
        super(MyWindow,self).__init__()

        self.setWindowTitle("ProgramNameToBeDecided")
        self.setFixedSize(900,600)
        font = QtGui.QFont("Arial", 12)
        self.setFont(font)

        self.stackedWidget = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stackedWidget)

        self.setupWidget = SetupWidget(self)
        self.initalGuessWidget = InitialGuessWidget(self)
        self.constituentCountingWidget = ConstituentCountingWidget(self)

        self.stackedWidget.addWidget(self.setupWidget)
        self.stackedWidget.addWidget(self.initalGuessWidget)
        self.stackedWidget.addWidget(self.constituentCountingWidget)
        
        self.stackedWidget.setCurrentIndex(0)
        
        self.show()

        self.stackedWidget.setFocus(QtCore.Qt.NoFocusReason)

    def MoveToSetupWidget(self, p_st=None, lowerCL=None, upperCL=None):
        if p_st is not None:
            self.setupWidget.AddResultsToTable(p_st, lowerCL, upperCL)
        self.setupWidget.Clear()
        self.stackedWidget.setCurrentIndex(0)

    def MoveToInitialGuessWidget(self):
        # Gather data
        imagePath = self.setupWidget.imagePathBox.text()
        
        if self.setupWidget.selectFullImageButton.isChecked():
            countAreaType = "Full"
        elif self.setupWidget.selectRectCropButton.isChecked():
            countAreaType = "Rectangular"
        elif self.setupWidget.selectCircCropButton.isChecked():
            countAreaType = "Circular"
        elif self.setupWidget.selectAnnularCropButton.isChecked():
            countAreaType = "Annular"
        
        # value will be none, array of length 3 (circ crop), or array of length 4 (rect or annular crop)
        countAreaBounds = self.setupWidget.countAreaBounds

        numStrata = 16
        
        # Initialize the initial guess widget
        self.initalGuessWidget.ReadImage(imagePath, numStrata, countAreaType, countAreaBounds)

        # change active widget
        self.stackedWidget.setCurrentIndex(1)

    def MoveToConstituentCountWidget(self):
        # Get required values
        if self.setupWidget.selectFullImageButton.isChecked():
            countAreaType = "Full"
        elif self.setupWidget.selectRectCropButton.isChecked():
            countAreaType = "Rectangular"
        elif self.setupWidget.selectCircCropButton.isChecked():
            countAreaType = "Circular"
        elif self.setupWidget.selectAnnularCropButton.isChecked():
            countAreaType = "Annular"

        # value will be none, array of length 3 (circ crop), or array of length 4 (rect or annular crop)
        countAreaBounds = self.setupWidget.countAreaBounds

        allocationStrategy = self.setupWidget.GetAllocationStrategy()
        initialGuesses = np.array(self.initalGuessWidget.initialGuesses)
        confidence = self.setupWidget.GetConfidence()
        moe = self.setupWidget.GetMOE()
        imagePath = self.setupWidget.imagePathBox.text()

        # Initialize widget
        self.constituentCountingWidget.InitializeCounting(initialGuesses, imagePath, allocationStrategy, countAreaType, countAreaBounds, confidence, moe)

        # Change active widget
        self.stackedWidget.setCurrentIndex(2)


def AutoAnalyzeSimImages():
    dir = "TestCases"
    files = os.listdir(dir)
    files.sort()
    numStrata_N = 3
    MOE = 0.01
    confidence = 0.95
    e_moe = 0.01

    guessValues = np.array([0.05, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 0.95])

    keyValues = []
    with open("key.csv", "r") as keyFile:
        reader = csv.reader(keyFile)
        for line in reader:
            temp = [line[0], line[1], line[2], line[3], line[4]]
            keyValues.append(temp)

    with open("SimOutput_1MOE_3x3.csv", 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for fileCount, file in enumerate(files):
            originalImage = io.imread(os.path.join(dir,file), as_gray=True)
            myMap = PixelMap(originalImage)

            N = myMap.numPixels
            N_h = int(N / numStrata_N**2)

            initialGuesses = []

            for i in range(numStrata_N):
                topBound = int(i * myMap.rows / numStrata_N)
                bottomBound = int((i+1) * myMap.rows/numStrata_N)
                for j in range(numStrata_N):
                    leftBound = int(j * myMap.cols / numStrata_N)
                    rightBound = int((j+1) * myMap.cols / numStrata_N)

                    selectedArea = myMap.originalImage[leftBound:rightBound+1, topBound:bottomBound+1]

                    numPorePixels = np.sum(np.where(selectedArea != 255, 1, 0))

                    porosity = numPorePixels / ((rightBound - leftBound) * (bottomBound - topBound))

                    closestGuess = guessValues[np.argmin(np.abs(porosity - guessValues))]

                    initialGuesses.append(closestGuess)
            
            initialGuesses = np.array(initialGuesses)
            
            initialStrataProportion = np.sum(initialGuesses) * N_h / N
            
            if initialStrataProportion > 0.5 and MOE > 1-initialStrataProportion:
                MOE = ((1-initialStrataProportion)+MOE) / 2 # want to keep +- MOE as close as possible on the open side. so if p=0.01, with 5% MOE, the CI should be (0,0.06)
                print(f"MOE stretches beyond range of [0,1] based on initial guess, reducing to {initialStrataProportion:.2f}")
            elif initialStrataProportion < 0.5 and MOE > initialStrataProportion:
                MOE = (initialStrataProportion + MOE) / 2
                print(f"MOE stretches beyond range of [0,1] based on initial guess, reducing to {initialStrataProportion:.2f}")
            
            upperCL = 1.0
            lowerCL = 0.0
            withinTolerance = False
            currentIter = 0
            maxIters = 1000
            currentIter = 0
            n = numStrata_N ** 2
            exponent = 1
            d = 0.5 * 0.75**(exponent-1) + 1
            lastGuess = -1
            while not withinTolerance and currentIter < maxIters:
                n = int(np.ceil(n))
                if n == lastGuess:
                    break
                lastGuess = n

                lowerCL, upperCL = scipy.stats.binom.interval(confidence, n, initialStrataProportion)
                lowerCL /= n
                upperCL /= n

                if ((upperCL - lowerCL) / 2) > MOE: # Eq.15 not satisfied
                    n *= d
                else: # Eq. 15 satisfied
                    pctDiff = abs((((upperCL - lowerCL) / 2) - MOE) / MOE)
                    if pctDiff > e_moe: # overestimating how many sample points needed
                        n /= d
                        exponent += 1
                        d = 0.5 * 0.75**(exponent-1) + 1
                    else:
                        withinTolerance = True
                currentIter += 1
            
            if currentIter == maxIters:
                raise(RuntimeError("Max Iterations reached"))

            # ###################################################### #
            # Allocate the total number of samples across the strata #
            # ###################################################### #

            def OptimalAllocation(n):
                W_h = 1 / numStrata_N**2
                
                optRatio = (W_h * (np.sum(np.sqrt(initialGuesses * (1-initialGuesses))))**2) / (np.sum(initialGuesses * (1-initialGuesses)))
                n *= optRatio

                strataVars = np.empty(numStrata_N**2)
                cnt = 0
                for i in range(numStrata_N):
                    topBound = int(i * myMap.rows/numStrata_N)
                    bottomBound = int((i+1)*myMap.rows/numStrata_N)
                    for j in range(numStrata_N):
                        leftBound = int(j*myMap.cols/numStrata_N)
                        rightBound = int((j+1)*myMap.cols/numStrata_N)
                        strataVars[cnt] = np.var(myMap.originalImage[topBound:bottomBound,leftBound:rightBound])
                        cnt += 1

                n_h = np.empty(numStrata_N**2)

                n_h = np.ceil(n * N_h * strataVars / np.sum(N_h * strataVars)).astype(int)

                # Ensure that at least one point sampled per strata
                n_h = np.maximum(n_h, 1)

                return n_h

            def ProportionalAllocation(n):
                n_h = np.ones(numStrata_N**2) * (n/numStrata_N**2)

                n_h = np.ceil(n_h).astype(int)

                return n_h

            n_h_opt = OptimalAllocation(n)

            n_h_prop = ProportionalAllocation(n)

            # ########################## #
            # Get pixel sample locations #
            # ########################## #

            csvRow = keyValues[fileCount]

            csvRow.append(f"{np.sum(n_h_opt) / np.sum(n_h_prop):.4f}")

            # ########################## #
            #   Optimal allocation       #
            # ########################## #

            p_h_opt = []
            cnt = 0
            for i in range(numStrata_N):
                topBound = int(i*myMap.rows/numStrata_N)
                bottomBound = int((i+1)*myMap.rows/numStrata_N)
                for j in range(numStrata_N):
                    leftBound = int(j*myMap.cols/numStrata_N)
                    rightBound = int((j+1)*myMap.cols/numStrata_N)

                    random = np.random.choice(np.arange(0,((bottomBound-topBound) * (rightBound-leftBound))), n_h_opt[cnt], replace=False)
                    random = np.array(np.unravel_index(random, (bottomBound-topBound,rightBound-leftBound)))
                    random[0,:] += topBound
                    random[1,:] += leftBound
            
                    pixels = list(zip(random[0,:], random[1,:]))
                    p_h = 0
                    for pixel in pixels:
                        if myMap.originalImage[pixel] != 255:
                            p_h += 1
                    
                    p_h_opt.append(p_h / len(pixels))

                    cnt += 1

            p_st_opt = np.sum(p_h_opt) * N_h / N
            p_h_opt = np.array(p_h_opt)

            lowerCL, upperCL = scipy.stats.binom.interval(confidence, np.sum(n_h_prop), p_st_opt) # Use n_h_prop to keep consistent MOE size
            lowerCL /= np.sum(n_h_prop)
            upperCL /= np.sum(n_h_prop)

            csvRow.append(np.sum(n_h_opt))
            csvRow.append(f"{p_st_opt*100:.3f}")
            csvRow.append(f"{lowerCL*100:.3f}")
            csvRow.append(f"{upperCL*100:.3f}")
            csvRow.append(f"{((upperCL-lowerCL)/2)*100:.3f}")

            # ########################## #
            #  Proportional allocation 1 #
            # ########################## #
            
            p_h_prop = []
            cnt = 0
            for i in range(numStrata_N):
                topBound = int(i*myMap.rows/numStrata_N)
                bottomBound = int((i+1)*myMap.rows/numStrata_N)
                for j in range(numStrata_N):
                    leftBound = int(j*myMap.cols/numStrata_N)
                    rightBound = int((j+1)*myMap.cols/numStrata_N)

                    random = np.random.choice(np.arange(0,((bottomBound-topBound) * (rightBound-leftBound))), n_h_prop[cnt], replace=False)
                    random = np.array(np.unravel_index(random, (bottomBound-topBound,rightBound-leftBound)))
                    random[0,:] += topBound
                    random[1,:] += leftBound

                    pixels = list(zip(random[0,:], random[1,:]))
                    p_h = 0
                    for pixel in pixels:
                        if myMap.originalImage[pixel] != 255:
                            p_h += 1
                    
                    p_h_prop.append(p_h / len(pixels))

                    cnt += 1
            
            p_st_prop = np.sum(p_h_prop) * N_h / N
            p_h_prop = np.array(p_h_prop)

            scipy.stats.binom.interval(confidence, np.sum(n_h_prop), p_st_prop)
            lowerCL, upperCL = scipy.stats.binom.interval(confidence, np.sum(n_h_prop), p_st_prop)
            lowerCL /= np.sum(n_h_prop)
            upperCL /= np.sum(n_h_prop)

            csvRow.append(np.sum(n_h_prop))
            csvRow.append(f"{p_st_prop*100:.3f}")
            csvRow.append(f"{lowerCL*100:.3f}")
            csvRow.append(f"{upperCL*100:.3f}")
            csvRow.append(f"{((upperCL-lowerCL)/2)*100:.3f}")

            # ########################## #
            #  Proportional allocation 2 #
            # ########################## #

            # # uses n_h_opt instead of n_prop to see if opt actually improves
            # new_prop_n_h = ProportionalAllocation(np.sum(n_h_opt))
            # p_h_prop = []
            # cnt = 0
            # for i in range(numStrata_N):
            #     topBound = int(i*myMap.rows/numStrata_N)
            #     bottomBound = int((i+1)*myMap.rows/numStrata_N)
            #     for j in range(numStrata_N):
            #         leftBound = int(j*myMap.cols/numStrata_N)
            #         rightBound = int((j+1)*myMap.cols/numStrata_N)

            #         random = np.random.choice(np.arange(0,((bottomBound-topBound) * (rightBound-leftBound))), new_prop_n_h[cnt], replace=False)
            #         random = np.array(np.unravel_index(random, (bottomBound-topBound,rightBound-leftBound)))
            #         random[0,:] += topBound
            #         random[1,:] += leftBound

            #         pixels = list(zip(random[0,:], random[1,:]))
            #         p_h = 0
            #         for pixel in pixels:
            #             if myMap.originalImage[pixel] != 255:
            #                 p_h += 1
                    
            #         p_h_prop.append(p_h / len(pixels))

            #         cnt += 1
            
            # p_st_prop = np.sum(p_h_prop) * N_h / N
            # p_h_prop = np.array(p_h_prop)

            # scipy.stats.binom.interval(confidence, np.sum(n_h_prop), p_st_prop)
            # lowerCL, upperCL = scipy.stats.binom.interval(confidence, np.sum(n_h_prop), p_st_prop)
            # lowerCL /= np.sum(n_h_prop)
            # upperCL /= np.sum(n_h_prop)

            # csvRow.append(np.sum(new_prop_n_h))
            # csvRow.append(f"{p_st_prop*100:.3f}")
            # csvRow.append(f"{lowerCL*100:.3f}")
            # csvRow.append(f"{upperCL*100:.3f}")
            # csvRow.append(f"{((upperCL-lowerCL)/2)*100:.3f}")

            writer.writerow(csvRow)
            csvFile.flush()

            # print()
            # print(f"Proportional ({np.sum(n_h_prop)} samples):")
            # print(f"{p_st_prop*100:.3f}, ({lowerCL*100:.3f}, {upperCL*100:.3f})")

def main():
    app = QtWidgets.QApplication(sys.argv)
    scipy.stats.binom.interval(0.95, 100, 10)
    app.setStyle("Fusion")

    win = MyWindow()

    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    # AutoAnalyzeSimImages()
