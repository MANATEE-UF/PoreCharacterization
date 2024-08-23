import numpy as np
import scipy.special
from skimage import io
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import os

# Class used to aid in displaying the image with grid overlayed onto sampled pixels
class PixelMap:
    def __init__(self,image:np.ndarray):
        self.rows = len(image)
        self.cols = len(image[0])
        self.numPixels = self.rows * self.cols
        self.originalImage = image

        if np.max(image) <= 1.0:
            image *= 255

    # Overlay a grid onto the original image and return centered around that grid
    def GetImageWithGridOverlay(self, pixelRow:int, pixelCol:int, newColor:tuple, numSurroundingPixels:int) -> np.ndarray: 
        # Center
        displayImage = self.originalImage

        displayImage[pixelRow][pixelCol] = newColor

        # Three above
        displayImage[pixelRow-1][pixelCol] = newColor
        displayImage[pixelRow-2][pixelCol] = newColor
        displayImage[pixelRow-3][pixelCol] = newColor

        # Three below
        displayImage[pixelRow+1][pixelCol] = newColor
        displayImage[pixelRow+2][pixelCol] = newColor
        displayImage[pixelRow+3][pixelCol] = newColor

        # Three right
        displayImage[pixelRow][pixelCol+1] = newColor
        displayImage[pixelRow][pixelCol+2] = newColor
        displayImage[pixelRow][pixelCol+3] = newColor

        # Three left
        displayImage[pixelRow][pixelCol-1] = newColor
        displayImage[pixelRow][pixelCol-2] = newColor
        displayImage[pixelRow][pixelCol-3] = newColor

        # Crop the image to center around the grid with numSurroundingPixels around
        displayImage = displayImage[pixelRow-numSurroundingPixels:pixelRow+numSurroundingPixels,pixelCol-numSurroundingPixels:pixelCol+numSurroundingPixels,:]

        return displayImage

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MyWindow(QMainWindow):

    def __init__(self, imagePath, numStrata_N, saveDir=None):
        super(MyWindow,self).__init__()

        self.saveDir = saveDir

        self.setWindowTitle("GASP")

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

        upText = QtWidgets.QLabel("Up Arrow Key For Re-sample")
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

        self.widget = QtWidgets.QWidget()
        self.widget.setLayout(vbox)
        self.setCentralWidget(self.widget)
        
        self.show()
        
        self.imageName = imagePath
        image = io.imread(imagePath, as_gray=True) # Enforces gray scale to ensure pixel map read consistently
        self.myMap = PixelMap(image)
        self.N = self.myMap.numPixels
        self.numStrata_N = numStrata_N
        self.N_h = int(self.N / self.numStrata_N**2)
        
        self.numSurroundingPixels = 50

        self.n_h = self.CalculateSampleSize()
        samplePositions, strataBounds = self.SampleImage(self.n_h)
        self.numGrids = np.sum(self.n_h)

        self.sampleIndex = 0
        self.strataIndex = 0

        self.poreData = np.zeros((self.numGrids))
        self.indexProgressText.setText(f"Sample: {self.sampleIndex}/{len(self.n_h[self.strataIndex])}, Strata: {self.strataIndex}/{self.numStrata_N**2}")

        displayImage = self.myMap.GetImageWithGridOverlay(0, 0, (1,1,1), self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(displayImage)
        self.sc.draw()

        self.widget.setFocus(QtCore.Qt.NoFocusReason)
    
    # Returns a list of optimal strata sample size (n_h)
    def CalculateSampleSize(self, alpha, MOE, e_moe, d) -> list:
        W_h = 1 / self.numStrata_N**2 # Value constant because image is split evenly. small differences neglected
        initialGuesses = np.ones(self.numStrata_N) * 0.5 # TODO: Replace this with AIVA
        initialStrataProportion = np.sum(self.numStrata_N * initialGuesses)
        variance = 1E6
        n_0 = np.sum(W_h * np.sqrt(initialGuesses * (1-initialGuesses)) / variance)
        n = n_0 / (1 + (np.sum(W_h * initialGuesses * (1-initialGuesses))) / (self.myMap.numPixels * variance))
        upperCL = self.UpperCL(n, int(np.ceil(initialStrataProportion * n)), alpha)
        lowerCL = self.LowerCL(n, int(np.ceil(initialStrataProportion * n)), n, alpha)


    # TODO: Should i be incrementing or decrementing? i.e., does higher A_U result in higher sum?
    def UpperCL(self, n, upperBound, alpha):
        # Pr(a,a' | A, A') = (A a) x (A' a') / (N n)
        probDistribution = 1.0
        A_U = upperBound # Ensures A_U always greater than i
        while probDistribution > alpha:
            A_U += 1
            sum = 0
            for i in range(upperBound):
                sum += scipy.special.comb(A_U, i, exact=True) * scipy.special.comb(self.N - A_U, n-i, exact=True) / scipy.special.comb(self.N, n, exact=True)
            
            probDistribution = sum
        
        return A_U

    def LowerCL(self, n, lowerBound, upperBound, alpha):
        probDistribution = 1.0
        A_L = upperBound
        while probDistribution > alpha:
            A_L += 1
            sum = 0
            for i in range(lowerBound, upperBound):
                sum += scipy.special.comb(A_L, i, exact=True) * scipy.special.comb(self.N - A_L, n-i, exact=True) / scipy.special.comb(self.N, n, exact=True)
            
            probDistribution = sum
    
    # Returns a list of pixel positions and a list indicating where strata index boundaries are
    def SampleImage(self, n_h):
        pass

    # Returns a (row, col) tuple corresponding to position of pixel in image for a sample and strata index
    def GetPixelInStrata(self,sampleIdx:int, strataIdx:int) -> tuple:
        pass

    def ZoomOut(self):
        if self.numSurroundingPixels < 300:
            self.numSurroundingPixels += 25
        
        newImage = self.myMap.GetGridCenterImage(self.poreIndex, self.gridIndex, self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.draw()

        if self.numSurroundingPixels >= 300:
            self.zoomOutButton.setEnabled(False)
        else:
            self.zoomOutButton.setEnabled(True)

        if self.numSurroundingPixels <= 25:
            self.zoomInButton.setEnabled(False)
        else:
            self.zoomInButton.setEnabled(True)
        
        self.widget.setFocus(QtCore.Qt.NoFocusReason)
    
    def ZoomIn(self):
        if self.numSurroundingPixels > 25:
            self.numSurroundingPixels -= 25
        
        newImage = self.myMap.GetGridCenterImage(self.poreIndex, self.gridIndex, self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.draw()

        if self.numSurroundingPixels == 25:
            self.zoomInButton.setEnabled(False)
        else:
            self.zoomInButton.setEnabled(True)

        if self.numSurroundingPixels == 300:
            self.zoomOutButton.setEnabled(False)
        else:
            self.zoomOutButton.setEnabled(True)
        
        self.widget.setFocus(QtCore.Qt.NoFocusReason)

    def keyPressEvent(self, event):
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
        else:
            pass
    
    def RecordDataPoint(self, value):
        # -1 is go back
        if value == -1 and self.gridIndex > 0:
            self.myMap.ChangeGridColor(self.poreIndex, self.gridIndex, (0,255,255))
            self.gridIndex -= 1
        elif value == -1 and self.gridIndex == 0 and self.poreIndex > 0:
            self.myMap.ChangeGridColor(self.poreIndex, self.gridIndex, (0,255,255))
            self.poreIndex -= 1
            self.gridIndex = self.numGrids-1
        elif value == -1 and self.gridIndex == 0 and self.poreIndex == 0:
            pass
        else:
            self.myMap.ChangeGridColor(self.poreIndex, self.gridIndex, (0,255,0))
            self.poreData[self.gridIndex][self.poreIndex] = value

            # At last grid index, move to next pore index
            if self.gridIndex == self.numGrids-1:
                self.gridIndex = 0
                self.poreIndex += 1            
            else:
                self.gridIndex += 1

        if self.poreIndex >= self.myMap.actualNumPoints:
            porosity = []
            if self.saveDir is None:
                saveFileName = f"{os.path.splitext(os.path.basename(self.imageName))[0]}_{len(self.myMap.gridCenters)}GridPoints_countData.csv"
            else:
                saveFileName = f"{self.saveDir}/{os.path.splitext(os.path.basename(self.imageName))[0]}_{len(self.myMap.gridCenters)}GridPoints_countData.csv"
            with open(saveFileName, "a") as f:
                for i, grid in enumerate(self.poreData):
                    porosity.append((np.sum(self.poreData[i]) / self.myMap.actualNumPoints))
                    np.savetxt(f, 
                        grid.reshape((len(self.myMap.gridCenters2D), len(self.myMap.gridCenters2D[0]))), 
                        fmt="%.2f", 
                        delimiter=",", 
                        header=f"{os.path.splitext(os.path.basename(self.imageName))[0]} with {len(self.myMap.gridCenters)} grid points \n Grid {i} \n",
                        footer=f"\n Porosity: {porosity[i] * 100}%")
                
                    print(f"Grid {i} Porosity: {porosity[i]*100:.2f}%")
            print(f"Average Porosity: {np.mean(porosity) * 100:.2f}")
            print(f"Standard Deviation: {np.std(porosity) * 100:.2f}")
            quit()

        newImage = self.myMap.GetGridCenterImage(self.poreIndex, self.gridIndex, self.numSurroundingPixels)
        self.indexProgressText.setText(f"Pore: {self.poreIndex+1}/{self.myMap.actualNumPoints}, Strata: {self.gridIndex+1}/{self.numGrids}")

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.draw()

def SingleImage(filename):
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    strataGrid_N = 4
    win = MyWindow(filename)

    win.show()
    sys.exit(app.exec_())

def Directory(dirname):
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    dirBaseName = os.path.basename(dirname)

    files = os.listdir(dirname)

    if not os.path.exists(f"{dirBaseName}_PROGRESS"):
        os.mkdir(f"{dirBaseName}_PROGRESS")
    else:
        lastFileAnalyzed = f"{os.listdir(dirBaseName)[-1].split('_')[0]}.tif"
        lastFileIdx = files.index(lastFileAnalyzed)
        files = files[lastFileIdx+1:]

    # Adjust the file name and number of grid points as needed
    numberOfGridPoints = 100
    numGrids = 3
    
    for file in files:
        SingleImage(f"{dirname}/{file}", numberOfGridPoints, numGrids)

def main():
    SingleImage("ManualCountTool/TestData/BSE_A_5kx_116.jpg")
    # Directory("/Users/mmika/Desktop/dividedImages")

if __name__ == "__main__":
    main()
