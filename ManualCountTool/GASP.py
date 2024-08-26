import numpy as np
import scipy.special
from skimage import io
from skimage.color import gray2rgb
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
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
        self.originalImage = image # grayscale

        if np.max(image) <= 1.0:
            self.originalImage *= 255
            self.originalImage = self.originalImage.astype(int)

    # Overlay a grid onto the original image and return centered around that grid
    def GetImageWithGridOverlay(self, pixelRow:int, pixelCol:int, newColor:tuple, numSurroundingPixels:int) -> np.ndarray: 
        # Center
        displayImage = gray2rgb(self.originalImage)

        displayImage[pixelRow][pixelCol] = newColor

        # Three above
        maxVal = 3 if pixelRow > 2 else pixelRow
        for i in range(1,maxVal):
            displayImage[pixelRow-i][pixelCol] = newColor

        # Three below
        maxVal = 3 if pixelRow < self.rows-3 else self.rows-pixelRow
        for i in range(1,maxVal):
            displayImage[pixelRow+i][pixelCol] = newColor

        # Three right
        maxVal = 3 if pixelCol < self.cols-3 else self.cols-pixelCol
        for i in range(1,maxVal):
            displayImage[pixelRow][pixelCol+i] = newColor

        # Three left
        maxVal = 3 if pixelCol > 2 else pixelCol
        for i in range(1,maxVal):
            displayImage[pixelRow][pixelCol-i] = newColor
        
        # pad image to ensure display proper
        displayImage = np.pad(displayImage, ((numSurroundingPixels+1, numSurroundingPixels+1), (numSurroundingPixels+1, numSurroundingPixels+1), (0,0)))

        # Crop the image to center around the grid with numSurroundingPixels around
        displayImage = displayImage[pixelRow:pixelRow+2*numSurroundingPixels,pixelCol:pixelCol+2*numSurroundingPixels,:]

        return displayImage

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MyWindow(QMainWindow):

    def __init__(self, imagePath, numStrata_N, alpha, MOE, e_moe, saveDir=None):
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

        self.widget = QtWidgets.QWidget()
        self.widget.setLayout(vbox)
        self.setCentralWidget(self.widget)
        
        self.show()
        
        self.imageName = imagePath
        image = io.imread(imagePath, as_gray=True)
        self.myMap = PixelMap(image)
        self.N = self.myMap.numPixels
        self.numStrata_N = numStrata_N
        self.N_h = int(self.N / self.numStrata_N**2)
        
        self.numSurroundingPixels = 50

        self.alpha = alpha
        self.MOE = MOE
        self.e_moe = e_moe
        self.d = 0.5

        self.n_h = self.CalculateSampleSize(self.alpha, self.MOE, self.e_moe, self.d)
        self.samplePositions = self.SampleImage(self.n_h)
        self.numGrids = np.sum(self.n_h)

        self.sampleIndex = 0
        self.strataIndex = 0
        self.gridIndex = 0 # Used to track flattend index, useful for writing to 1D poreData

        self.poreData = np.zeros((self.numGrids))
        self.indexProgressText.setText(f"Sample: {self.sampleIndex+1}/{self.n_h[self.strataIndex]}, Strata: {self.strataIndex+1}/{self.numStrata_N**2}")

        displayImage = self.myMap.GetImageWithGridOverlay(self.samplePositions[self.strataIndex][self.sampleIndex][0], self.samplePositions[self.strataIndex][self.sampleIndex][1], (50, 225, 248), self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(displayImage)
        self.sc.axes.set_yticks([])
        self.sc.axes.set_xticks([])
        self.sc.draw()

        self.widget.setFocus(QtCore.Qt.NoFocusReason)
    
    # Returns a list of optimal strata sample size (n_h)
    def CalculateSampleSize(self, alpha, MOE, e_moe, d) -> list:
        W_h = 1 / self.numStrata_N**2 # Value constant because image is split evenly. small differences neglected
        initialGuesses = np.ones(self.numStrata_N**2) * 0.2 # TODO: Replace this with AIVA
        initialStrataProportion = np.sum(initialGuesses) * self.N_h / self.N
        variance = np.sum(W_h * np.sqrt(initialGuesses * (1 - initialGuesses)))**2 / self.numStrata_N**2 - ((1/self.N) * np.sum(W_h * initialGuesses * (1 - initialGuesses))) # highest variance given the initial guesses, assuming only one point taken per stratum
        upperCL = 1.0
        lowerCL = 0.0
        withinTolerance = False
        while not withinTolerance:
            n = np.sum(W_h * np.sqrt(initialGuesses * (1-initialGuesses)) / variance)
            n = int(np.ceil(n))
            upperCL = self.UpperCL_A(n, initialStrataProportion, alpha) / n
            lowerCL = self.LowerCL_A(n, initialStrataProportion, alpha) / n

            if ((upperCL - lowerCL) / 2) > MOE: # Eq.15 not satisfied
                variance *= d
            else: # Eq. 15 satisfied
                pctDiff = abs((((upperCL - lowerCL) / 2) - MOE) / MOE)
                if pctDiff > e_moe: # variance too low, overestimating how many sample points needed
                    variance /= d
                    d += (1-d)/2
                    variance *= d
                else:
                    withinTolerance = True
            
            print(f"MOE:{((upperCL - lowerCL) / 2):.3f}")

        # n_h = self.ProportionalAllocation(n)
        n_h = self.OptimalAllocation(n)

        print(f"{np.sum(n_h)} samples needed to achieve {MOE} MOE with {100*(alpha)}% CI")

        print(n_h)

        return n_h

    def UpperCL_A(self, n, p, alpha):
        sum = 0
        A_U = 0
        while sum <= alpha + (1-alpha)/2:
            A_U += 1
            sum = 0
            for i in range(A_U):
                sum += scipy.special.comb(n, i, exact=True) * np.power(p,i) * np.power((1-p),n-i)
        
        return A_U-1

    def LowerCL_A(self, n, p, alpha):
        sum = 1.0
        A_L = 0
        while sum > alpha + (1-alpha)/2:
            A_L += 1
            sum = 0
            for i in range(A_L, n):
                sum += scipy.special.comb(n, i, exact=True) * np.power(p,i) * np.power((1-p),n-i)
        
        return A_L-1
    
    def OptimalAllocation(self, n):
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

        return n_h

    def ProportionalAllocation(self, n):
        n_h = np.ones(self.numStrata_N**2) * (n/self.numStrata_N**2)

        n_h = np.ceil(n_h).astype(int)

        return n_h

    # Returns a 2D list of pixel positions
    # First axis is strata axis, second axis is sample axis
    def SampleImage(self, n_h):
        pixels = []
        cnt = 0
        for i in range(self.numStrata_N):
            topBound = int(i*self.myMap.rows/self.numStrata_N)
            bottomBound = int((i+1)*self.myMap.rows/self.numStrata_N)
            for j in range(self.numStrata_N):
                leftBound = int(j*self.myMap.cols/self.numStrata_N)
                rightBound = int((j+1)*self.myMap.cols/self.numStrata_N)

                randomY = np.random.choice(np.arange(topBound, bottomBound), n_h[cnt], replace=False)
                randomX = np.random.choice(np.arange(leftBound, rightBound), n_h[cnt], replace=False)

                pixels.append(list(zip(randomY, randomX)))

                cnt += 1
        
        return pixels

    def ZoomOut(self):
        if self.numSurroundingPixels < 300:
            self.numSurroundingPixels += 25
        
        newImage = self.myMap.GetImageWithGridOverlay(self.samplePositions[self.strataIndex][self.sampleIndex][0], self.samplePositions[self.strataIndex][self.sampleIndex][1], (50, 225, 248), self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.axes.set_yticks([])
        self.sc.axes.set_xticks([])
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
        
        newImage = self.myMap.GetImageWithGridOverlay(self.samplePositions[self.strataIndex][self.sampleIndex][0], self.samplePositions[self.strataIndex][self.sampleIndex][1], (50, 225, 248), self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.axes.set_yticks([])
        self.sc.axes.set_xticks([])
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
            if self.saveDir is None:
                saveFileName = f"{os.path.splitext(os.path.basename(self.imageName))[0]}_{self.numGrids}GridPoints_countData.csv"
            else:
                saveFileName = f"{self.saveDir}/{os.path.splitext(os.path.basename(self.imageName))[0]}_{self.numGrids}GridPoints_countData.csv"
            with open(saveFileName, "a") as f:
                for i, n in enumerate(self.n_h):
                    if i == 0:
                        bounds = [0, n]
                    else:
                        bounds = [np.cumsum(self.n_h[:i])[-1], np.cumsum(self.n_h[:i])[-1] + n]
                    p_h.append(np.average(self.poreData[bounds[0]:bounds[1]]))

                p_st = np.sum(p_h) * self.N_h / self.N

                # variance = (1 / self.N)**2 * np.sum((self.N_h**2 * (self.N_h-self.n_h) * p_h * (1-p_h)) / ((self.N_h-1) * self.n_h))

                upperCL = self.UpperCL_A(self.numGrids, p_st, self.alpha) / self.numGrids
                lowerCL = self.LowerCL_A(self.numGrids, p_st, self.alpha) / self.numGrids

                np.savetxt(f, 
                    p_h, 
                    fmt="%.2f", 
                    delimiter=",", 
                    header=f"{os.path.splitext(os.path.basename(self.imageName))[0]} with {self.numGrids} samples across {self.numStrata_N**2} strata. \n alpha = {self.alpha} \n MOE = {self.MOE} \n e_moe = {self.e_moe} \n d = {self.d}",
                    footer=f"\n Porosity: {p_st * 100}% \n {self.alpha}% CI: ({lowerCL, upperCL}) \n MOE: {(upperCL - lowerCL) / 2}")
                
            print(f"\n Porosity: {p_st * 100:.3f}% \n {100*(self.alpha)}% CI: ({lowerCL:.3f}, {upperCL:.3f}) \n MOE: {(upperCL - lowerCL) / 2:.3f}")
            quit()

        newImage = self.myMap.GetImageWithGridOverlay(self.samplePositions[self.strataIndex][self.sampleIndex][0], self.samplePositions[self.strataIndex][self.sampleIndex][1], (50, 225, 248), self.numSurroundingPixels)
        self.indexProgressText.setText(f"Sample: {self.sampleIndex+1}/{self.n_h[self.strataIndex]}, Strata: {self.strataIndex+1}/{self.numStrata_N**2}")

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.axes.set_yticks([])
        self.sc.axes.set_xticks([])
        self.sc.draw()

def SingleImage(filename):
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    strataGrid_N = 4
    win = MyWindow(filename, strataGrid_N, alpha=0.95, MOE=0.05, e_moe=0.05)

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
    SingleImage("./ManualCountTool/simPores_21PctPorosity.png")
    # Directory("/Users/mmika/Desktop/dividedImages")

if __name__ == "__main__":
    main()
